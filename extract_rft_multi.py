import pandas as pd
import re
import jsonlines
import json
from openai import OpenAI
with open("sys_eval_prompt.txt","r",encoding="utf-8") as f:
    eval_prompt = f.read()

api_key = ""
client = OpenAI(api_key = api_key, base_url= "https://api.openai.com/v1/")

def get_response(prompt , model = "gpt-4.1", temperature = 0):
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
    answer = get_response(prompt)
    if "yes" in answer:
        return 1
    else:
        return 0

def normalize(text):
    text = text.upper().strip()
    text = re.sub(r'[^a-zA-Z0-9]', '', text)
    return text

data_path = "train.parquet"
data_df = pd.read_parquet(data_path)
gt_answer = dict()
for i, row in data_df.iterrows():
    prompt = row['prompt']
    question = row["extra_info"]["question"]
    gt = row["extra_info"]["selected_answer"]
    # gt_answer[question] = normalize(gt)
    gt_answer[question] = gt

gen_file = 'data/self_2700_5000_7b_click_4.jsonl'
with jsonlines.open(gen_file) as reader:
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
        # answer = normalize(answer)

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
        # if ground_truth in ans or ''.join(sorted(ground_truth)) == ''.join(sorted(ans)):
        if same(key, ground_truth, ans):
            flag += 1
            # tmp_step += steps
            if len(steps) > max_step:
                max_step = len(steps)
                tmp_step = steps
    if flag != 0 and flag != len(val):
        succ += 1
    else:
        tmp_step = []
    res += tmp_step
print(succ)

with open('data/corr_data_self_2700_5000_7b_click_4_all.jsonl', 'a', encoding='utf-8') as f:
    for item in res:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")