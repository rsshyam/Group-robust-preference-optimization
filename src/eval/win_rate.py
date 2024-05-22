import sys
from pathlib import Path
import pandas as pd
import logging
from tqdm import tqdm
import pickle
from fast_oai import call_chats####openai_api added here, check file before using
sys.path.append('..')
sys.path.append('path')#folderpath --- src or not?
import os
print(sys.path)
#from epinet import get_shuffle_iterator
system_prompt = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Output your final verdict by strictly following this format: 'A' if assistant A is better, 'B' if assistant B is better, and 'C' for a tie. Output only that character and do not include any other characters or spaces."

user_prompt = "[User Question]\n{prompt}\n[The Start of Assistant A's Answer]\n{sample1}\n[The End of Assistant A's Answer]\n[The Start of Assistant B's Answer]\n{sample2}\n[The End of Assistant B's Answer]\n"


def get_user_prompt(row):
    prompt = row["prompt"]
    sample1 = row['sample_only']
    sample2 = row['correct response']
    return user_prompt.format(prompt=prompt, sample1=sample1, sample2=sample2)

def main(csv_dir_path, overwrite_model_result=False):
    csv_dir_path = Path(csv_dir_path)

    for csv_path in tqdm(csv_dir_path.iterdir()):
        print(f'processing {csv_path}')
        if not str(csv_path).endswith('.csv'):
            continue
        df=pd.read_csv(csv_path)
        print(df.columns)
        if df.columns[0]!='step':#in case the csv file directly starts with data
            df = pd.read_csv(csv_path,header=None)
            columns=["step", "prompt", "sample","correct response"]#change depending on file creation
            df.columns=columns
            print(df.columns)
        print(df.head)
        if 'model_result' in df.columns and not overwrite_model_result:
            mask = ~df['model_result'].isin(['Win', 'Lose', 'Tie'])
        else:
            mask = pd.Series(True, index=df.index)
        #print(df.columns) 
        #keys = list(dataset.keys())
        df['sample_only'] = df.apply(lambda row: row['sample'][len(row['prompt']):], axis=1)##removes prompt part of the sample and stores it in sample_only
        #df['sft_target'] = df.apply(lambda row: dataset[row['prompt']]['correct response'], axis=1)

        df['user_prompt'] = df.apply(get_user_prompt, axis=1)
        user_prompt_list = df.loc[mask, 'user_prompt'].tolist()
        system_prompt_gen = (system_prompt for _ in range(len(user_prompt_list)))
        completions = call_chats(zip(system_prompt_gen, user_prompt_list))
        vals = []
        for i, dec in enumerate(completions):
            if dec == 'A':
                vals.append("Win")
            elif dec == 'B':
                vals.append("Lose")
            elif dec == 'C':
                vals.append('Tie')
            else:
                logging.warning(f"Unexpected decision {dec} on row {i}")
                vals.append(dec)
        df.loc[mask, 'model_result'] = vals
        print(f'writing to {csv_path}')
        df.to_csv(csv_path, index=False)

if __name__ == '__main__':
    main(*sys.argv[1:])