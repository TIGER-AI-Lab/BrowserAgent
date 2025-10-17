import pandas as pd
import re
import jsonlines
import json
import argparse

parser = argparse.ArgumentParser(description="Run multi-turn response generation with customizable file paths.")
parser.add_argument('--gen_file', type=str, 
                    default='', 
                    help='Path to the gen_file')
parser.add_argument('--output_file', type=str, 
                    default='./test.jsonl', 
                    help='Output file path for writing the data (e.g., /path/to/output.jsonl)')
args = parser.parse_args()



with open("system_prompt_with_history_info.txt","r",encoding = "utf-8") as f:
    system_prompt = f.read()

user_prompt = """
Objective: {}
Observation: {}
HISTORY_ACTION: {}
HISTORY_info: {}
"""

with jsonlines.open(args.gen_file) as reader:
    gen_data = list(reader)

res = list()
for data in gen_data:
    input = data['input_seq']
    output = data['output_seq']

    question = re.findall(r'Objective: (.*?)\nObservation', input)[0]
    # url = re.findall(r'URL:(.*?)\nObservation', input)[0]
    obs = input.split('Observation: ')[1].split('\nHISTORY_ACTION:')[0]
    his = input.split('HISTORY_ACTION:')[1].split('\n\n')[0]
    his_info = input.split('HISTORY_info:')[1].split('\n\n')[0]

    user = user_prompt.format(question, obs, his, his_info)

    message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user}, {'role': 'assistant', 'content': output}]

    messages = {'messages': message}
    res.append(messages)

with open(args.output_file, 'a', encoding='utf-8') as f:
    for item in res:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")