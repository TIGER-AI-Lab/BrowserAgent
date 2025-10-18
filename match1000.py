import pandas as pd
import re
import jsonlines


def normalize(text):
    text = text.upper().strip()
    text = re.sub(r'[^a-zA-Z0-9]', '', text)
    return text


data_path=""


data_df = pd.read_parquet(data_path)
data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)
data_df = data_df[:1000]
gt_answer = dict()
for i, row in data_df.iterrows():
    prompt = row['prompt']
    question = row["extra_info"]["question"]
    gt = row["extra_info"]["selected_answer"]
    gt_answer[question] = normalize(gt)


gen_file = 'hotpot.jsonl'
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
    if question not in gt_answer:
        continue

    if not re.findall(r"```(.*?)```", output):
        answer = " "
    else:
        answer = re.findall(r"```(.*?)```", output)[0]

    if 'stop' in answer:
        try:
            ans = normalize(re.findall(r"\[(.*?)\]", answer)[0])
        except:
            ans = ""
        ground_truth = gt_answer[question]
        if ground_truth in ans or ''.join(sorted(ground_truth)) == ''.join(sorted(ans)):
            suc += 1
            steps += data['trajectory_length']
    else:
        emp += 1

print(f"问题数目：{len(gt_answer)}")
print(f"回答正确数目：{suc}")
print(f"正确率：{suc/len(gt_answer)}")
print(f"未回答数目：{emp}")

print(f"平均步数：{steps / suc}")
