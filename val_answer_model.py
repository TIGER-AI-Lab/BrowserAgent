import pandas as pd
import re
import jsonlines
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



data_path = ""
data_df = pd.read_parquet(data_path)
gt_answer = dict()
for i, row in data_df.iterrows():
    prompt = row['prompt']
    question = row["extra_info"]["question"]
    gt = row["extra_info"]["selected_answer"]
    gt_answer[question] = gt

gen_file = 'nq.jsonl'
with jsonlines.open(gen_file) as reader:   
    gen_data = list(reader)

steps = 0
suc = 0
emp = 0
for data in gen_data:
    content = data['trajectory']
    input = content[-1]['input_seq']
    output = content[-1]['output_seq']

    question = re.findall(r'Objective: (.*?)\nObservation', input)[0]

    if not re.findall(r"```(.*?)```", output):
        answer = " "
    else:
        answer = re.findall(r"```(.*?)```", output)[0]
    
    if 'stop' in answer:
        try:
            ans = re.findall(r"\[(.*?)\]", answer)[0]
        except:
            ans = ""
        ground_truth = gt_answer[question]
        if same(question, ground_truth, ans):
            suc += 1
            steps += data['trajectory_length']
    else:
        emp += 1


print(f"问题数目：{len(gen_data)}")
print(f"回答正确数目：{suc}")
print(f"未回答数目：{emp}")
print(f"平均步数：{steps/suc}")
