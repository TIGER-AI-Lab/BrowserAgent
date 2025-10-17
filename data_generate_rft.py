import json
import pandas as pd
from openai import OpenAI
import threading
import json
from typing import List, Dict, Any
import requests
import argparse



lock=threading.Lock()
api_key = "sk-proj-1234567890"
client = OpenAI(api_key = api_key, base_url= "http://localhost:5001/v1")
with open("system_prompt_with_history_info.txt","r",encoding = "utf-8") as f:
    system_prompt = f.read()

def call_tool_server(trajectory_ids: List[str], actions: List[str], finish: List[bool], **kwargs: Dict[str, List[Any]]) -> Dict[str, Any]:
    """querying the tool server for the observation and done flag using aiohttp"""
    env_url = "http://0.0.0.0:30810/get_observation"
    server_url = env_url
    # prepare payload
    extra_fields = [{
        "url": (
            "https://tigerai.ca/wiki/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
        )
    }]
    data = {
        "trajectory_ids": trajectory_ids,
        "actions": actions,
        "finish": finish,
        "extra_fields": extra_fields
    }
    
    try:
        resp = requests.post(env_url, json=data, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

user_prompt = """
Objective: {}
Observation: {}
HISTORY_ACTION: {}
HISTORY_info: {}
"""

def get_response(prompt , model = "qwen2.5-7b", temperature = 0):
    response = client.chat.completions.create(
        model = model,
        messages = [{"role": "user", "content": prompt}],
        temperature = temperature,
        max_tokens = 1024
    )
    model_answer = response.choices[0].message.content
    return model_answer

import re
def extract_command(text):
    blocks = re.findall(r'```\s*([^\s].*?[^\s])\s*```', text, re.DOTALL)
    
    if not blocks:
        return " "

    last_command = blocks[-1].strip()
    last_command = last_command.replace("```","")
    return last_command.strip()

def extract_conclusion(text):
    blocks = re.findall(r'<conclusion>\s*(.*?)\s*</conclusion>', text, re.DOTALL)

    if not blocks:
        return " "

    last_conclusion = blocks[-1].strip()
    return last_conclusion

def write_a_data(output_file, action_list):
    lock.acquire()
    for written_data in action_list:
        with open(output_file,"a",encoding = "utf-8") as fw:
            fw.write(json.dumps(written_data,ensure_ascii=False) + "\n")
    lock.release()


import uuid

def Get_multi_turn_response(question, answer, output_file):
    tar_id = str(uuid.uuid4())
    history = "\n"
    obj = question
    history_info = "\n"
    action_list=[]
    try:
        jsoned_data = call_tool_server([tar_id],[''],[False])
        obs = jsoned_data['observations'][0]
    except Exception as e:
        print(e)

    for i in range(30):
        try:
            obs = obs.split('Observation:\n')[1].split('\nParsed Previous Action:')[0]
        except:
            pass
        real_prompt = user_prompt.format(obj,obs,history,history_info)
        prompt = system_prompt + "\n\n" + real_prompt
        response = get_response(prompt, temperature=1)
        last_command = extract_command(response)
        last_info = extract_conclusion(response)

        history = history + last_command + "\n"
        history_info = history_info + last_info + "\n"

        try:
            jsoned_data = call_tool_server([tar_id],[response],[False])
            obs = jsoned_data['observations'][0]
        except Exception as e:
            print(e)
        action_list.append({"input_seq":prompt, "output_seq":response})

        if "stop" in last_command:
            write_a_data(output_file, action_list)
            call_tool_server([tar_id],[response],[True])
            return
    write_a_data(output_file, action_list)


max_threads = 16
number_to_process = 20000
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-turn response generation with customizable file paths.")
    parser.add_argument('--output_file', type=str, 
                        default='./test.jsonl', 
                        help='Output file path for writing the data (e.g., /path/to/output.jsonl)')
    parser.add_argument('--data_path', type=str, 
                        default='', 
                        help='Path to the data file (e.g., /path/to/train.parquet)')
    args = parser.parse_args()
    print(f"Output file: {args.output_file}")
    print(f"Data file: {args.data_path}")

    data_df = pd.read_parquet(args.data_path)
    data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)
    # data_df = data_df[:1000] 
    cnt = 0
    threads = []
    for i, row in data_df.iterrows():

        cnt += 1
        # if cnt < 10000:
        #     continue
        
        question = row["extra_info"]["question"]
        gt = row["extra_info"]["selected_answer"]
        for i in range(4):
            t = threading.Thread(target=Get_multi_turn_response, args=(question, gt, args.output_file))
            threads.append(t)
        
        if cnt % max_threads == 0 or cnt == number_to_process:
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            threads = []

        if cnt == number_to_process:
            break