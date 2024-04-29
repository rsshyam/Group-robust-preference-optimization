import datasets
import torch
import ast
import json
from torch.utils.data import DataLoader, Dataset
from src.utils import get_local_dir, TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import pandas as pd
import ast
import matplotlib.pyplot as plt

COUNTRIES=[
    'Nigeria',
    'Egypt',
    'India (Current national sample)',
    'China',
    'Japan',
    'Germany',
    'France',
    'Spain',
    'United States',
    'Canada',
    'Brazil',
    'Argentina',
    'Australia',
    'New Zealand'
]

def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    if name == 'shp':
        data = get_shp(split, silent=silent, cache_dir=cache_dir)
    elif name == 'hh':
        data = get_hh(split, silent=silent, cache_dir=cache_dir)
    elif name == 'se':
        data = get_se(split, silent=silent, cache_dir=cache_dir)
    elif name == 'jeopardy':
        data = get_jeopardy(split, silent=silent, cache_dir=cache_dir)
    elif "jeopardy" in name:
        value = name.split("_")[-1]
        if value!='final':
            value=int(value)
        data = get_jeopardy_value(split, value, silent=silent, cache_dir=cache_dir)
    elif name.startswith('oqa'):
        if name == 'oqa':
            data = get_oqa(split, silent=silent, cache_dir=cache_dir)
        else:
            name, attribute, group = name.split('_')
            # should be something like oqa_SEX_male
            data = get_oqa_group(split, attribute, group, silent=silent, cache_dir=cache_dir)
    elif name == 'hel':
        data = get_hel(split, silent=silent, cache_dir=cache_dir)
    elif name == 'helon':
        data = get_helon(split, silent=silent, cache_dir=cache_dir)
    elif name == 'helrej':
        data = get_helrej(split, silent=silent, cache_dir=cache_dir)
    elif name == 'heltot':
        data = get_heltot(split, silent=silent, cache_dir=cache_dir)
    elif name == 'har':
        data = get_har(split, silent=silent, cache_dir=cache_dir)
    elif 'reddit' in name:
        group_id = int(name.split('_')[-1])
        split = 'validation' if split == 'test' else split
        data = get_reddit(split, group_id, silent=silent, cache_dir=cache_dir)
    elif 'hel_' in name:
        change_split=name.split('_')[-1]
        if 'train' in split:
            split=f'train[:{change_split}%]'
        #else:
            #split=f'test[:{change_split}%]'
        data = get_hel(split, silent=silent, cache_dir=cache_dir)
    elif name == 'GOqa':
        data=get_goqa(split, silent=silent, cache_dir=cache_dir)
    elif name == 'GOqMa':
        data=get_goqa_multiple(split, silent=silent, cache_dir=cache_dir)
    elif 'GOqa' in name:
        group_id=int(name.split('_')[-1])
        data=get_goqa_group(split, group_id, silent=silent, cache_dir=cache_dir)
    elif 'GOqMa' in name:
        group_id=int(name.split('_')[-1])
        data=get_goqa_group_multiple(split, group_id, silent=silent, cache_dir=cache_dir)
    
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    assert set(list(data.values())[0].keys()) == {'responses', 'pairs', 'sft_target'}, \
        f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"

    return data

def get_oqa(
        split: str, 
        attribute: str,
        group: str, 
        multi_pair: bool=False, 
        n_pairs: int=4, 
        silent: bool=False, 
        plot_distr: bool = False, 
        cache_dir: str = None
    ) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    # TODO: cache_dir is unused currently
    # TODO: better abstraction  of group to other types except SEX

    OQA_ATTRIBUTES = ['SEX']
    OQA_GROUPS = ['Male','Female']

    ATTRIBUTE = attribute #OQA_ATTRIBUTES[group_id[0]]
    GROUP = group #OQA_GROUPS[group_id[1]]
    
    if split not in ('test', 'train'):
        raise ValueError(f'split {split} not recognized (valid: test, train)')
    print(f'Loading GPO (OQA) dataset from file...\n')
    df = pd.read_csv(f'data/{split}_oqa.csv')    
    
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
    def make_prompt_and_responses(elt):
        # take a row of the csv and return a prompt and responses
        question = elt['question']
        options = ast.literal_eval(elt['options'])
        options = [opt for opt in options if opt != "Refused"]
        attribute = elt['attribute']
        group = elt['group']
        distribution = elt['D_H']
        numbers_str = distribution.strip('[]').split()
        numbers_float = [float(x) for x in numbers_str]
        distribution = np.array(numbers_float)
        
        prompt = f"Answer the following question as if you were {attribute} of {group}: {question}\nRespond with a single letter:"
        letters_opt = letters[:len(options)]
        for opt, letter in zip(options, letters_opt):
            prompt += f"\n{letter}. {opt}"
        responses = [letter for letter in letters_opt]

        ranks = np.argsort(distribution)
        if multi_pair is True:
            # pairs given as (correct, wrong) based on explicit user preference (deterministic)
            pairs = [tuple(sorted([ranks[i], ranks[j]],reverse=True)) for i in range(len(ranks)) for j in range(i)]
            pairs = random.sample(pairs,min(n_pairs,len(pairs)))
        else:
            # single pair (correct,wrong) is the best-preferred (correct) vs least-preferred (wrong)
            correct_response_index = np.where(distribution==max(distribution))
            wrong_response_index = np.where(distribution==min(distribution))
            pairs = [(correct_response_index,wrong_response_index)]

        sft_target = options[np.max(ranks)] # best-preferred option
        return prompt, dict(responses=responses, pairs=pairs, sft_target=sft_target)
    
    def plot_distribution(all_data: Dict[str, Dict]):
        correct_idx = []
        wrong_idx = []
        for prompt in all_data:
            for pair in data[prompt]['pairs']:
                correct_idx.append(pair[0])
                wrong_idx.append(pair[1])
        plt.figure()
        plt.bar(np.arange(len(correct_idx)), height=correct_idx, label='correct')
        plt.bar(np.arange(len(wrong_idx)), height=wrong_idx, label='wrong')
        plt.legend()
        plt.savefig(f'./dataload_plt/oqa_distribution_{ATTRIBUTE}_{GROUP}.png')

    all_data = {}
    for idx, row in tqdm.tqdm(df.iterrows(), disable=silent, desc="Processing OQA"):
        if row['attribute'] == ATTRIBUTE and row['group'] == GROUP:
            prompt, data  = make_prompt_and_responses(row)
            all_data[prompt] = data
    
    if plot_distr is True:
        plot_distribution(all_data)

    return all_data

def get_oqa_group(split: str, attribute: str, group: str, silent: bool=False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    if split not in ('test', 'train'):
        raise ValueError(f'split {split} not recognized (valid: test, train)')
    groups = pd.read_csv('data/groups.csv')
    # Check if the pair exists in the DataFrame
    if not ((groups['attribute'] == attribute) & (groups['group'] == group)).any():
        raise ValueError(f"The pair attribute={attribute}, group={group} is not present in the DataFrame.")
    print(f'Loading GPO dataset from file...')
    df = pd.read_csv(f'data/{split}_oqa.csv')
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    def make_prompt_and_responses(elt):
        # take a row of the csv and return a prompt and responses
        question = elt['question']
        options = ast.literal_eval(elt['options'])
        options = [opt for opt in options if opt != "Refused"]
        this_attribute = elt['attribute']
        this_group = elt['group']
        if this_attribute != attribute or this_group != group:
            return None, None
        distribution = elt['D_H']
        numbers_str = distribution.strip('[]').split()
        numbers_float = [float(x) for x in numbers_str]
        distribution = np.array(numbers_float)
        prompt = f"Answer the following question as if you were {attribute} of {group}: {question}\nRespond with a single letter:"
        for opt, letter in zip(options, letters):
            prompt += f"\n{letter}. {opt}"
        responses = [letter for letter in letters[:len(options)]]
        ranks = np.argsort(distribution)
        pairs = [(ranks[i], ranks[j]) for i in range(len(ranks)) for j in range(i)]
        sft_target = responses[ranks[-1]]
        return prompt, dict(responses=responses, pairs=pairs, sft_target=sft_target)

    all_data = {}
    for idx, row in tqdm.tqdm(df.iterrows(), disable=silent, desc="Processing OQA"):
        prompt, data  = make_prompt_and_responses(row)
        all_data[prompt] = data
    return all_data




def get_jeopardy_value(split: str, value: int, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    if split not in ('test', 'train'):
        raise ValueError(f'split {split} not recognized (valid: test, train)')
    if value not in (200, 400, 600, 800, 1000, 1200, 1600, 2000, 'dd', 'final'):
        raise ValueError(f"Jeopardy! dataset requested with value {value} that isn't present")
    print(f'Loading Jeopardy! dataset from file...')
    with open(f'data/{split}_jeopardy_data.json', 'r') as f:
        data = json.load(f)
    '''
    data is of the form

    {'category': 'HISTORY', 'air_date': '2004-12-31', 'question': "'For the last 8 years of his life, Galileo was under house arrest for espousing this man's theory'", 'value': '$200', 'answer': 'Copernicus', 'round': 'Jeopardy!', 'show_number': '4680', 'wrong_answer': 'Kepler'}
    '''
    # TODO: will need to iterate on prompts to some extent
    def make_prompt_and_responses(elt):
        category = elt['category']
        question = elt['question']
        if elt['value'] is None:
            elt_value = 'final'
        elif elt['value'] not in (200, 400, 600, 800, 1000, 1200, 1600, 2000):
            elt_value = 'dd'
        else:
            elt_value = int(elt['value'].replace("$", "").replace(",", ""))
        if elt_value != value:
            return None, None
        answer = elt['answer']
        wrong_answer = elt['wrong_answer']
        prompt = f'{category}, for {value}: {question}'
        # change null token to empty string
        # responses = [answer, 'null', wrong_answer]
        responses = [answer, "", wrong_answer]
        pairs = [(0, 1), (0, 2), (1, 2)]
        # take a single sample
        pairs = [random.choice(pairs)]
        return prompt, dict(responses=responses, pairs=pairs, sft_target=answer)
    all_data = {}
    for row in tqdm.tqdm(data, desc="Processing Jeopardy!", disable=silent):
        prompt, data = make_prompt_and_responses(row)
        if prompt is None:
            continue
        all_data[prompt] = data
    return all_data


def get_jeopardy(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    if split not in ('test', 'train'):
        raise ValueError(f'split {split} not recognized (valid: test, train)')
    print(f'Loading Jeopardy! dataset from file...')
    with open(f'data/{split}_jeopardy_data.json', 'r') as f:
        data = json.load(f)
    '''
    data is of the form

    {'category': 'HISTORY', 'air_date': '2004-12-31', 'question': "'For the last 8 years of his life, Galileo was under house arrest for espousing this man's theory'", 'value': '$200', 'answer': 'Copernicus', 'round': 'Jeopardy!', 'show_number': '4680', 'wrong_answer': 'Kepler'}
    '''
    # TODO: will need to iterate on prompts to some extent
    def make_prompt_and_responses(elt):
        category = elt['category']
        question = elt['question']
        value = elt['value']
        answer = elt['answer']
        wrong_answer = elt['wrong_answer']
        prompt = f'{category}, for {value}: {question}'
        # change null token to empty string
        # responses = [answer, 'null', wrong_answer]
        responses = [answer, "", wrong_answer]
        pairs = [(0, 1), (0, 2), (1, 2)]
        # take a single sample
        pairs = [random.choice(pairs)]
        return prompt, dict(responses=responses, pairs=pairs, sft_target=answer)
    all_data = {}
    for row in tqdm.tqdm(data, desc="Processing Jeopardy!", disable=silent):
        prompt, data = make_prompt_and_responses(row)
        all_data[prompt] = data
    return all_data

def get_reddit(split: str, group_id, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    print(f'Loading Reddit TL;DR dataset ({split} split, group_id {group_id}) from Huggingface...')
    dataset = datasets.load_dataset('openai/summarize_from_feedback', 'comparisons', split=split, cache_dir=cache_dir)

    def split_prompt_and_responses(ex):
        prompt = ex['info']['post']
        chosen_response = ex['summaries'][ex['choice']]['text']
        rejected_response = ex['summaries'][1 - ex['choice']]['text']
        group = ex['info']['subreddit']
        return prompt, chosen_response, rejected_response, group

    data = defaultdict(lambda: defaultdict(list))
    datapoint_per_group = {'relationships': 52346, 'AskReddit': 12963, 'weddingplanning': 644, 'jobs': 782, 'dating_advice': 2257, 'legaladvice': 1769, 'askwomenadvice': 536, 'offmychest': 1447, 'personalfinance': 1900, 'relationship_advice': 7037, 'loseit': 1015, 'needadvice': 305, 'BreakUps': 622, 'GetMotivated': 253, 'Advice': 1802, 'self': 1402, 'Dogtraining': 381, 'pettyrevenge': 580, 'college': 209, 'cats': 300, 'dogs': 482, 'travel': 319, 'books': 190, 'AskDocs': 331, 'Parenting': 356, 'running': 363, 'Cooking': 161, 'Pets': 205, 'tifu': 1901}
    # {52346: 'relationships', 12963: 'AskReddit', 7037: 'relationship_advice', 2257: 'dating_advice', 1901: 'tifu', 1900: 'personalfinance', 1802: 'Advice', 1769: 'legaladvice', 1447: 'offmychest', 1402: 'self', 1015: 'loseit'}
    group_ids = {0: 'relationships', 1: 'AskReddit', 2: 'relationship_advice', 3: 'dating_advice', 4: 'tifu', 5: 'personalfinance', 6: 'Advice', 7: 'legaladvice', 8: 'offmychest', 9: 'self', 10: 'loseit'}
    uniq_topic = {}
    count=0
    count_2=0
    for row in tqdm.tqdm(dataset, desc='Processing Reddit', disable=silent):
        prompt, chosen, rejected, group = split_prompt_and_responses(row)
        #print(group)
        if group_ids[group_id] != group:
            count=count+1
            continue

        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        if prompt in data:
            count_2+=1
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen
        if group not in uniq_topic:
            uniq_topic[group] = 0
        uniq_topic[group] += 1
    print(count)
    print(count_2)
    print(len(data))
    return data


def get_hh_datasets(split: str, variants: list, silent: bool = False, cache_dir: str = None) -> dict:
    """
    Load and merge specific variants of the Anthropic Helpful-Harmless dataset from Huggingface.

    Parameters:
        split (str): Dataset split (e.g., 'train', 'test').
        variants (list): List of dataset variants to load (e.g., ['helpful-base', 'helpful-online', 'helpful-rejection-sampled']).
        silent (bool): If True, suppress tqdm progress display.
        cache_dir (str): Directory for caching the dataset, optional.

    Returns:
        dict: A structured dictionary with the combined dataset content formatted for model training or evaluation.
    """
    def extract_anthropic_prompt(text):
        """Utility function to extract the prompt part from a response."""
        return text.split('\n\nAssistant:')[0] + '\n\nAssistant:'

    def split_prompt_and_responses(ex):
        """Splits the dataset entry into prompt and responses."""
        prompt = extract_anthropic_prompt(ex['chosen'])
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: {'responses': [], 'pairs': [], 'sft_target': ''})
    for variant in variants:
        print(f'Loading {variant} dataset ({split} split) from Huggingface...')
        dataset = datasets.load_dataset('Anthropic/hh-rlhf', data_dir=variant, split=split, cache_dir=cache_dir)
        print('done')

        for row in tqdm.tqdm(dataset, desc=f'Processing {variant}', disable=silent):
            prompt, chosen, rejected = split_prompt_and_responses(row)
            responses = [chosen, rejected]
            n_responses = len(data[prompt]['responses'])
            data[prompt]['pairs'].append((n_responses, n_responses + 1))
            data[prompt]['responses'].extend(responses)
            # Set the sft_target only if not already set (keep the first chosen response encountered)
            if not data[prompt]['sft_target']:
                data[prompt]['sft_target'] = chosen

    return data


def main():
    data = get_oqa('train', 'SEX', 'Male', plot_distr=True)
    #Example of using the function to load and merge three datasets
    
    #data = load_and_merge_hh_datasets('train', ['helpful-rejection-sampled', 'helpful-online', 'helpful-base'])

    # data = get_oqa('train')
    # data = get_jeopardy_value('train', 200)
    # data = get_jeopardy_value('train', 'final')


if __name__ == "__main__":
    main()