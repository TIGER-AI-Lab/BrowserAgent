import json
import pandas as pd
from openai import OpenAI
import threading
import json
from typing import List, Dict, Any
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


lock=threading.Lock()
api_key = "sk-proj-1234567890"
client = OpenAI(api_key = api_key, base_url= "http://localhost:5001/v1")
with open("system_prompt_click_nourl.txt","r",encoding = "utf-8") as f:
    system_prompt = f.read()

def call_tool_server(trajectory_ids: List[str], actions: List[str], finish: List[bool], **kwargs: Dict[str, List[Any]]) -> Dict[str, Any]:
    """querying the tool server for the observation and done flag using aiohttp"""
    env_url = "http://localhost:30810/get_observation"
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
        resp = requests.post(env_url, json=data, timeout=1200)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

user_prompt = """
Objective: {}
Observation: {}
HISTORY_ACTION: {}
"""

def get_response(prompt , model = "qwen2.5-7b", temperature = 0):
    #print(prompt)
    # try:
    response = client.chat.completions.create(
        model = model,
        messages = [{"role": "user", "content": prompt}],
        temperature = temperature,
        max_tokens = 1024
    )
    # print(response)
    # exit(0)
    model_answer = response.choices[0].message.content
    # except:
    #     model_answer = "```error```"
    return model_answer

import re
def extract_command(text):
    #print(text)
    blocks = re.findall(r'```\s*([^\s].*?[^\s])\s*```', text, re.DOTALL)
    
    if not blocks:
        return " "

    last_command = blocks[-1].strip()
    last_command = last_command.replace("```","")
    return last_command.strip()


def generate_filename():
    now = datetime.now()
    return f"{now.strftime('%Y%m%d_%H%M%S')}_webarena_results.jsonl"


global_filename = None

def write_a_data(action_list, filename=None):
    global global_filename
    if filename is None:
        if global_filename is None:
            global_filename = generate_filename()
        filename = global_filename

    
    trajectory_data = {
        "trajectory": action_list,
        "trajectory_length": len(action_list)
    }
    
    lock.acquire()
    with open(filename, "a", encoding="utf-8") as fw:
        fw.write(json.dumps(trajectory_data, ensure_ascii=False) + "\n")
    lock.release()


import uuid

def Get_multi_turn_response(question, answer):
    tar_id = str(uuid.uuid4())
    history = "\n"
    obj = question
    # url = init_url
    history_info = "\n"
    action_list = []
    is_error = False
    error_msg = ""
    
    try:
        
        jsoned_data = call_tool_server([tar_id], [''], [False])
        obs = jsoned_data['observations'][0]
        
        for i in range(30):
            try:
                obs = obs.split('Observation:\n')[1].split('\nParsed Previous Action:')[0]
            except:
                pass
            
            real_prompt = user_prompt.format(obj, obs, history)
            prompt = system_prompt + "\n\n" + real_prompt
            
            try:
                response = get_response(prompt, temperature=1)
                last_command = extract_command(response)
                
                history = history + last_command + "\n"
                
                action_list.append({"input_seq": prompt, "output_seq": response})
                
                
                jsoned_data = call_tool_server([tar_id], [response], [False])
                obs = jsoned_data['observations'][0]
                
                if "stop" in last_command:
                    
                    call_tool_server([tar_id], [response], [True])
                    break
                    
            except Exception as e:
                is_error = True
                error_msg = str(e)
                print(f" {i}, {e}")
                break
                
    except Exception as e:
        raise e
        is_error = True
        error_msg = str(e)
        print(f"{e}")
    
    
    if action_list:
        action_list[-1]["is_error"] = is_error
        action_list[-1]["error_msg"] = error_msg
    else:
        
        action_list.append({
            "input_seq": f"question: {question}",
            "output_seq": "error",
            "is_error": is_error,
            "error_msg": error_msg
        })
    
    
    write_a_data(action_list)

# data_path="" 

max_threads = 64  
number_to_process = 99999

def process_single_item(row):
    
    question = row["extra_info"]["question"]
    gt = row["extra_info"]["selected_answer"]
    return Get_multi_turn_response(question, gt)

if __name__ == "__main__":
    data_df = pd.read_parquet(data_path)
    data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)
    data_df = data_df[:1000]
    
   
    data_to_process = data_df.head(number_to_process)
    
    print(f" {len(data_to_process)} , {max_threads}")
    
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
       
        future_to_row = {
            executor.submit(process_single_item, row): idx 
            for idx, row in data_to_process.iterrows()
        }
        
        
        completed_count = 0
        for future in as_completed(future_to_row):
            idx = future_to_row[future]
            try:
                result = future.result()  
                completed_count += 1
                if completed_count % 10 == 0:  
                    print(f" {completed_count}/{len(data_to_process)} ")
            except Exception as e:
                print(f" {idx} ,{e}")
                completed_count += 1
    
    print(f" {completed_count}")
