import pandas as pd
import re
import jsonlines

def normalize(text):
    text = text.upper().strip()
    text = re.sub(r'[^a-zA-Z0-9]', '', text)
    return text

# data_path="/home/yutao/dataset/wiki_hotpotqa_new/data/dev-00000-of-00001.parquet" # hotpot
# data_path="/home/zhiheng/WikiRL/ragen/env/wiki/data/puzzle/test.parquet" # nq
# data_path="/home/yutao/dataset/wiki_triviaqa/data/dev-00000-of-00001.parquet" # tri
data_path="/home/yutao/dataset/wiki_data/musique/dev.parquet" # mus
# data_path="/home/yutao/dataset/wiki_data/bamboogle/test.parquet" # bam
# data_path="/home/yutao/dataset/wiki_data/2wiki/dev.parquet" # 2wiki
# data_path="/home/yutao/dataset/wiki_data/popqa/test.parquet" # pop

data_df = pd.read_parquet(data_path)
gt_answer = dict()
for i, row in data_df.iterrows():
    prompt = row['prompt']
    question = row["extra_info"]["question"]
    gt = row["extra_info"]["selected_answer"]
    gt_answer[question] = normalize(gt)

gen_file = '/home/yutao/brosweragent/mini_webarena/mus_qwen_t0.jsonl'
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
            ans = normalize(re.findall(r"\[(.*?)\]", answer)[0])
        except:
            ans = ""
        ground_truth = gt_answer[question]
        if ground_truth in ans or ''.join(sorted(ground_truth)) == ''.join(sorted(ans)):
            suc += 1
            steps += data['trajectory_length']
    else:
        emp += 1


print(f"问题数目：{len(gen_data)}")
print(f"回答正确数目：{suc}")
print(f"正确率：{suc/len(gen_data)}")
print(f"未回答数目：{emp}")
print(f"平均步数：{steps/suc}")