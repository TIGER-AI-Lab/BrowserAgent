import pandas as pd
import re
import jsonlines
import json
from openai import OpenAI
import argparse

parser = argparse.ArgumentParser(description="Run multi-turn response generation with customizable file paths.")
parser.add_argument('--data_path', type=str, 
                    default='', 
                    help='Path to the data file (e.g., /path/to/train.parquet)')
parser.add_argument('--gen_file', type=str, 
                    default='', 
                    help='Path to the gen_file')
parser.add_argument('--output_file', type=str, 
                    default='./test.jsonl', 
                    help='Output file path for writing the data (e.g., /path/to/output.jsonl)')
args = parser.parse_args()



with open("sys_eval_prompt.txt","r",encoding="utf-8") as f:
    eval_prompt = f.read()

def normalize(text):
    text = text.upper().strip()
    text = re.sub(r'[^a-zA-Z0-9]', '', text)
    return text


data_df = pd.read_parquet(args.data_path)
gt_answer = dict()
for i, row in data_df.iterrows():
    prompt = row['prompt']
    question = row["extra_info"]["question"]
    gt = row["extra_info"]["selected_answer"]
    gt_answer[question] = normalize(gt)


with jsonlines.open(args.gen_file) as reader:
    gen_data = list(reader)

gen_data_dict = dict()
pre_q = re.findall(r'Objective: (.*?)\nObservation', gen_data[0]['input_seq'])[0]
tmp = list()
for data in gen_data:
    tmp.append(data)
    input = data['input_seq']
    output = data['output_seq']

    question = re.findall(r'Objective: (.*?)\nObservation', input)[0]

    if pre_q != question:
        pre_q = question
        tmp = [data]

    if not re.findall(r"```(.*?)```", output):
        answer = " "
    else:
        answer = re.findall(r"```(.*?)```", output)[0]

    if question not in gen_data_dict:
        gen_data_dict[question] = list()

    if "stop" in answer:
        answer = re.findall(r"\[(.*?)\]", answer)[0]
        answer = normalize(answer)

        gen_data_dict[question].append({'answer': answer, 'steps': tmp})
        tmp = list()

res = list()
succ = 0
for key, val in gen_data_dict.items():
    ground_truth = gt_answer[key]
    max_step = 1
    for data in val:
        ans = data['answer']
        steps = data['steps']
        if ground_truth in ans or ''.join(sorted(ground_truth)) == ''.join(sorted(ans)):
            res += steps
            succ += 1
print(succ)

with open(args.output_file, 'a', encoding='utf-8') as f:
    for item in res:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")