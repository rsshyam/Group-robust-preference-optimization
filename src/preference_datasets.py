import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from src.utils import get_local_dir, TemporarilySeededRandom
from src.groupstuff.group_dataset import GroupDataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
from src.groupstuff.data_processing import get_oqa,get_oqa_group,get_hh_datasets
from src.groupstuff.global_opinion_data_processing import get_goqa

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]


def strip_html_tags(html_string):
    """Strip HTML tags from a string, except for <code> tags (which contain real code in the StackExchange answers)."""
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_string, 'html.parser')

    # Initialize an empty list to store the text
    text = []
    for element in soup.children:
        if isinstance(element, NavigableString):
            continue
        if element.name == 'p':
            text.append(''.join(child.string for child in element.children if isinstance(child, NavigableString)))
        elif element.name == 'pre':
            for code in element.find_all('code'):
                text.append("<code>" + code.get_text() + "</code>")
        elif element.name == 'code':
            text.append("<code>" + element.get_text() + "</code>")

    # Join the text together with newlines in between
    text = "\n\n".join(text)

    return text


def get_se(split, silent=False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the StackExchange dataset from Huggingface, and return a dict of prompts and responses. See get_hh for the format.
    
       We strip the HTML tags from the responses (except for <code> tags), and we add necessary newlines.
    """
    print(f'Loading SE dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('HuggingFaceH4/stack-exchange-preferences', cache_dir=cache_dir)['train']
    print('done')

    # shuffle the dataset and select 1% for test
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(int(len(dataset) * 0.01))) if split == 'test' else dataset.select(
        range(int(len(dataset) * 0.01), len(dataset)))

    def strip_html(x):
        x['question'] = strip_html_tags(x['question'])
        for a in x['answers']:
            a['text'] = strip_html_tags(a['text'])
        return x

    dataset = dataset.map(strip_html, num_proc=64)

    data = defaultdict(dict)
    for row in tqdm.tqdm(dataset, desc='Processing SE', disable=silent):
        prompt = '\n\nHuman: ' + row['question'] + '\n\nAssistant:'
        responses = [' ' + a['text'] for a in row['answers']]
        scores = [a['pm_score'] for a in row['answers']]

        pairs = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                pairs.append((i, j) if scores[i] > scores[j] else (j, i))

        data[prompt]['responses'] = responses
        data[prompt]['pairs'] = pairs
        data[prompt]['sft_target'] = max(responses, key=lambda x: scores[responses.index(x)])

    return data

def get_shp(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Stanford Human Preferences dataset from Huggingface and convert it to the necessary format. See hh for the format.

       We filter preference pairs to only keep pairs where the score ratio is at least 2.
       For this dataset, the sft_target is the response with the highest score.
    """
    print(f'Loading SHP dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('stanfordnlp/SHP', split=split, cache_dir=cache_dir)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing SHP', disable=silent):
        prompt = '\n\nHuman: ' + row['history'] + '\n\nAssistant:'
        responses = [' ' + row['human_ref_A'], ' ' + row['human_ref_B']]
        scores = [row['score_A'], row['score_B']]
        if prompt in data:
            n_responses = len(data[prompt]['responses'])
        else:
            n_responses = 0
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])
        if score_ratio < 2:
            continue

        # according to https://huggingface.co/datasets/stanfordnlp/SHP
        data[prompt]['pairs'].append((n_responses, n_responses + 1) if row['labels'] == 1 else (n_responses + 1, n_responses))
        data[prompt]['responses'].extend(responses)
        data[prompt]['scores'].extend(scores)

    for prompt in data:
        data[prompt]['sft_target'] = max(data[prompt]['responses'], key=lambda x: data[prompt]['scores'][data[prompt]['responses'].index(x)])
        del data[prompt]['scores']

    return data


def get_hh(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.
    
       The dataset is converted to a dictionary with the following structure:
       {
           'prompt1': {
               'responses': List[str],
               'pairs': List[Tuple[int, int]],
               'sft_target': str
           },
           'prompt2': {
               ...
           },
       }

       Prompts should be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
       
       For this dataset, the sft_target is just the chosen response.
    """
    print(f'Loading HH dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, cache_dir=cache_dir)
    print('done')

    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex['chosen'])
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing HH', disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen

    return data


def get_dataset(name: str, split: str, train_frac: float = 0.8, silent: bool = False, cache_dir: str = None, test:bool = False):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    if name == 'shp':
        data = get_shp(split=split, silent=silent, cache_dir=cache_dir)
    elif name == 'hh':
        data = get_hh(split=split, silent=silent, cache_dir=cache_dir)
    elif name == 'se':
        data = get_se(split=split, silent=silent, cache_dir=cache_dir)
    elif 'goqma' in name:
        group_id=int(name.split('_')[-1])
        data=get_goqa(split=split,train_frac= train_frac,group_id= group_id,multi_pair=True, silent=silent, cache_dir=cache_dir)
    elif 'goqa' in name:
        group_id=int(name.split('_')[-1])
        data=get_goqa(split=split,train_frac=train_frac,group_id= group_id,multi_pair=False, silent=silent, cache_dir=cache_dir)
    elif name in ['hel','helon','helraj','har']:
        data = get_hh_datasets(split, variant=[name], silent=silent, cache_dir=cache_dir)
    elif name == 'heltot':
        data = get_hh_datasets(split, variant=['hel','helon','helraj'], silent=silent, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    assert set(list(data.values())[0].keys()) == {'responses', 'pairs', 'sft_target'}, \
        f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"
        
        #If test mode reduce the dataset size to only 10 datapoints
    if test:
        print('Using test data config')        
        data_keys, new_data = list(data.keys())[:32], dict()
        for key in data_keys:
            new_data[key] = data[key]
        data = new_data
        print('Pruned test data config')

    return data

def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.
    
       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""
    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    return collate_fn


def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int, group: int=None) -> Dict:
    """Tokenize a single batch element.

       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected
    #print('element',group)
    #print('element wth group')
    if group is not None:
        batch['group']=group

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch


def process_dataset(dataset: Dict, 
                    truncation_mode: str, 
                    sep_pairs: bool, 
                    unique_prompts: bool,
                    group_handling: bool = False, 
                    group_id: Optional[int] = None):
    """
    Process the dataset to prepare data for batching, considering separation of pairs and group handling.

    Args:
        dataset: The dataset to process.
        truncation_mode: Truncation mode to apply ('keep_start' or 'keep_end').
        sep_pairs: Whether to separate pairs into individual items.
        group_handling: Whether group-specific logic is enabled.
        group_id: Optional identifier for the data group.

    Returns:
        A list of processed dataset items ready for batching.
    """
    flat_data = []
    for prompt, data in dataset.items():
        # Process based on separation of pairs and group handling
        if len(data['pairs']) > 1 and sep_pairs:
            for pair_index, pair in enumerate(data['pairs']):
                responses = [data['responses'][pair[0]], data['responses'][pair[1]]]
                # Include group_id if group_handling is True
                data_tuple = (prompt, responses, [(0, 1)], data['sft_target'], truncation_mode)
                if group_handling:
                    flat_data.append((*data_tuple, group_id))
                    #print((*data_tuple, group_id))
                else:
                    flat_data.append(data_tuple)
                if unique_prompts:
                    break
        else:
            data_tuple = (prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode)
            if group_handling:
                flat_data.append((*data_tuple, group_id))
                #print((*data_tuple, group_id))
            else:
                flat_data.append(data_tuple)
    return flat_data

def transform_weighted_item(prompt, responses, pairs, sft_target, truncation_mode, group_id):
    """
    Example transformation function for items from a weighted iterable,
    adjusting them to match the structure expected by the processing logic.
    """
    # Transform the item to match the expected structure (prompt, responses, pairs, etc.)
    # This is placeholder logic and should be replaced with actual transformation code.
    prompt=prompt[0]#to remove tuple
    #print(prompt,'prompt')
    tresponses=[]
    for r in responses:
        tresponses.append(r[0])#to remove tuple
    responses=tresponses
    tpairs=[]
    for p in pairs:
        tpairs.append(tuple(q.item() for q in p))#to remove tensored versions
    pairs=tpairs
    sft_target=sft_target[0]#to remove tuple
    #print(sft_target)
    truncation_mode=truncation_mode[0]#to remove tuple
    #
    #print(truncation_mode)
    group_id=group_id.item()
    return prompt, responses, pairs, sft_target, truncation_mode, group_id

def process_batches(flat_data, batch_size, collate_fn, tokenizer, max_length, max_prompt_length, sft_mode, n_examples,n_epochs, split, silent,shuffle, permutation_seeds, unique_prompts, group_handling, weighted,n_groups):
    """
    Processes data into batches and yields them. Handles both weighted and non-weighted scenarios.

    Args:
        flat_data: The preprocessed data ready for batching.
        batch_size: The size of each batch.
        collate_fn: Function to collate data points into batches.
        tokenizer: The tokenizer to use for processing text.
        max_length: Maximum sequence length.
        max_prompt_length: Maximum prompt length.
        sft_mode: Whether to use SFT mode for tokenization.
        n_examples: The number of examples to process. If None, processes all examples.
        split: The data split being used (e.g., 'train', 'test').
        silent: If True, does not print progress messages.
        is_train: Indicates if the current processing is for training data.
        weighted: Indicates if weighted sampling should be used.
    """
    epoch_idx = 0
    example_idx = 0
    done = False
    while not done:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_examples} examples on {split} split')
            break
        # Shuffle data if required

        if shuffle:
            print(next(permutation_seeds),'next seed')
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        if group_handling and weighted:
            # Replace with your actual weighted sampling logic
            # For example, use a DataLoader with a WeightedRandomSampler
            iterable = GroupDataset(flat_data,n_groups).get_loader()  # Assuming this is a DataLoader
        else:
            iterable = flat_data

        batch = []
        # Process each data point into a batch
        for data_point in iterable:
            if done:
                break

            prompt, responses, pairs, sft_target, truncation_mode = data_point[:5]
            #print(data_point)
            group_id = data_point[5] if group_handling else None

            # Adjust for weighted scenario unpacking
            if group_handling and weighted:
                # Example transformation function to match the non-weighted structure
                prompt, responses, pairs, sft_target, truncation_mode, group_id = transform_weighted_item(prompt, responses, pairs, sft_target, truncation_mode, group_id)
            # Specific logic for SFT mode
            if sft_mode:
                batch_element = tokenize_batch_element(prompt, sft_target, sft_target, truncation_mode, tokenizer, max_length, max_prompt_length, group=group_id if group_handling else None)
                batch_element = {k: v for k, v in batch_element.items() if 'rejected' not in k}
                batch.append(batch_element)
                example_idx += 1
                if len(batch) == batch_size:
                    #print(batch)
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f'Finished generating {n_examples} examples on {split} split')
                        done = True
                    batch = []
            else:
                # Standard processing
                for p in pairs:
                    if done:
                        break
                    batch_element = tokenize_batch_element(prompt, responses[p[0]], responses[p[1]], truncation_mode, tokenizer, max_length, max_prompt_length, group=group_id if group_handling else None)
                    batch.append(batch_element)
                    example_idx += 1
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f'Finished generating {n_examples} examples on {split} split')
                            done = True
                        batch = []
                    if unique_prompts:
                        break

        if done:
            break

        epoch_idx += 1



def get_batch_iterator(names: List[str],
                       tokenizer,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool = True,
                       max_length: int = 512,
                       max_prompt_length: int = 128,
                       sft_mode: bool = False,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed: int = 0,
                       silent: bool = False,
                       cache_dir: Optional[str] = None,
                       test_dataset: bool = False,
                       group_handling: bool = False,
                       train_frac: float=0.8,
                       sep_pairs: bool = False,
                       weighted: bool = False,
                       mode: str = 'batch_iterator') -> Iterator[Dict]:
    """Get an iterator over batches of data with optional group handling. 
    Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
        test_dataset: Flag to indicate if using a test dataset.
        group_handling: Flag to enable group-based data handling.
        sep_paris: Flag to enable separation of response pairs corresponding to a single prompt
    """

    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()
    if 'gen' in split:
        split=split.split('_')[0]
        unique_prompts=True
    else:
        unique_prompts=False
    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        group_counts=[]
        for name in names:
            truncation_mode = 'keep_end' if name in ['hh', 'har', 'hel', 'helon', 'helrej', 'heltot'] else 'keep_start'
            if mode=='batch_iterator':
                dataset = get_dataset(name=name, train_frac=train_frac, split=split, silent=silent, cache_dir=cache_dir, test=test_dataset)
                group_id = names.index(name) if group_handling else None
                flat_data.extend(process_dataset(dataset, truncation_mode, sep_pairs,unique_prompts,group_handling,group_id))
            elif mode=='count_groups':
                g_len=0
                for prompt, data in get_dataset(name=name, train_frac=train_frac, split=split, silent=silent, cache_dir=cache_dir, test=test_dataset).items():
                    g_len+= 1 if unique_prompts else len(data['pairs'])
                group_counts.append(g_len)
    
    if mode=='count_groups':
        return group_counts

    collate_fn = get_collate_fn(tokenizer)
    n_groups=len(names)
    return process_batches(flat_data, batch_size, collate_fn, tokenizer, max_length, max_prompt_length, sft_mode, n_examples, n_epochs, split, silent, shuffle, permutation_seeds, unique_prompts, group_handling, weighted,n_groups)

    
def strings_match_up_to_spaces(str_a: str, str_b: str) -> bool:
    """Returns True if str_a and str_b match up to spaces, False otherwise."""
    for idx in range(min(len(str_a), len(str_b)) - 2):
        if str_a[idx] != str_b[idx]:
            if str_a[idx] != ' ' and str_b[idx] != ' ':
                return False
            else:
                if str_a[idx] == ' ':
                    str_a = str_a[:idx] + str_a[idx + 1:]
                else:
                    str_b = str_b[:idx] + str_b[idx + 1:]

    return True