import pandas as pd
import re
import jsonlines
import json
import os
from openai import OpenAI
import argparse

os.environ['OPENAI_API_KEY'] = ""


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

client = OpenAI(base_url= "https://api.openai.com/v1/")

def get_response(prompt , model="gpt-4.1", temperature = 0):
    #print(prompt)
    # try:
    response = client.chat.completions.create(
        model = model,
        messages = [{"role": "user", "content": prompt}],
        temperature = temperature,
        max_tokens = 1024
    )
    model_answer = response.choices[0].message.content
    # except:
    #     model_answer = "```error```"
    return model_answer


def same(question, gt, ans):
    prompt = eval_prompt.format(question, gt, ans)
    # gpt
    answer_gpt = get_response(prompt, model="gpt-4.1")
    # gemini
    answer_gemini = get_response(prompt, model="gemini-2.5-flash")
    # claude
    answer_claude = get_response(prompt, model="claude-4-opus-20250514")

    if "yes" in answer_gpt and "yes" in answer_gemini and "yes" in answer_claude:
        return 1
    else:
        return 0


data_df = pd.read_parquet(args.data_path)
gt_answer = dict()
for i, row in data_df.iterrows():
    prompt = row['prompt']
    question = row["extra_info"]["question"]
    gt = row["extra_info"]["selected_answer"]
    gt_answer[question] = gt


with jsonlines.open(args.gen_file) as reader:
    gen_data = list(reader)
print(gen_data[0])
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

        gen_data_dict[question].append({'answer': answer, 'steps': tmp})
        tmp = list()

res = list()
succ = 0
for key, val in gen_data_dict.items():
    flag = 0
    ground_truth = gt_answer[key]
    max_step = 1
    tmp_step = list()
    for data in val:
        ans = data['answer']
        steps = data['steps']
        if same(key, ground_truth, ans):
            flag += 1
            # tmp_step += steps
            if len(steps) > max_step:
                max_step = len(steps)
                tmp_step = steps
    if (flag != 0 and flag != len(val)) or flag == 1:
        succ += 1
    else:
        tmp_step = []
    res += tmp_step
print(succ)

with open(args.output_file, 'a', encoding='utf-8') as f:
    for item in res:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
