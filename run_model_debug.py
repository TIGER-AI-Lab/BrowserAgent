import json
import pandas as pd
from openai import OpenAI
import threading
import json
from typing import List, Dict, Any
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

lock=threading.Lock()
api_key = "sk-proj-1234567890"
client = OpenAI(api_key = api_key, base_url= "http://localhost:5005/v1")
with open("/home/yutao/brosweragent/mini_webarena/system_prompt_with_history_info.txt","r",encoding = "utf-8") as f:
    system_prompt = f.read()

def call_tool_server(trajectory_ids: List[str], actions: List[str], finish: List[bool], **kwargs: Dict[str, List[Any]]) -> Dict[str, Any]:
    """querying the tool server for the observation and done flag using aiohttp"""
    env_url = "http://localhost:30812/get_observation"
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
HISTORY_info: {}
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

def extract_conclusion(text):
    # 匹配所有 <conclusion>...</conclusion> 的内容
    blocks = re.findall(r'<conclusion>\s*(.*?)\s*</conclusion>', text, re.DOTALL)

    if not blocks:
        return " "

    last_conclusion = blocks[-1].strip()
    return last_conclusion


def generate_filename():
    now = datetime.now()
    return f"{now.strftime('%Y%m%d_%H%M%S')}_webarena_results_debug.jsonl"

# 全局文件名，在程序开始时初始化
global_filename = None

def write_a_data(action_list, filename=None):
    global global_filename
    if filename is None:
        if global_filename is None:
            global_filename = generate_filename()
        filename = global_filename
    
    # 将整个轨迹作为一个JSON对象
    trajectory_data = {
        "trajectory": action_list,
        "trajectory_length": len(action_list)
    }
    
    lock.acquire()
    with open(filename, "a", encoding="utf-8") as fw:
        fw.write(json.dumps(trajectory_data, ensure_ascii=False) + "\n")
    lock.release()
# def write_a_data(input,output):
#     written_data = {"input_seq":input,"output_seq":output}
#     lock.acquire()
#     with open("tmp.jsonl","a",encoding = "utf-8") as fw:
#         fw.write(json.dumps(written_data,ensure_ascii=False) + "\n")
#     lock.release()

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
        # 初始化环境
        jsoned_data = call_tool_server([tar_id], [''], [False])
        obs = jsoned_data['observations'][0]
        
        for i in range(30):
            try:
                obs = obs.split('Observation:\n')[1].split('\nParsed Previous Action:')[0]
            except:
                pass
            
            real_prompt = user_prompt.format(obj, obs, history, history_info)
            prompt = system_prompt + "\n\n" + real_prompt
            
            try:
                response = get_response(prompt, temperature=1)
                last_command = extract_command(response)
                last_info = extract_conclusion(response)
                
                history = history + last_command + "\n"
                history_info = history_info + last_info + "\n"
                
                action_list.append({"input_seq": prompt, "output_seq": response})
                
                # 调用工具服务器
                jsoned_data = call_tool_server([tar_id], [response], [False])
                obs = jsoned_data['observations'][0]
                
                if "stop" in last_command:
                    # 结束轨迹
                    call_tool_server([tar_id], [response], [True])
                    break
                    
            except Exception as e:
                is_error = True
                error_msg = str(e)
                print(f"步骤 {i} 出现错误: {e}")
                break
                
    except Exception as e:
        raise e
        is_error = True
        error_msg = str(e)
        print(f"初始化或整体执行出现错误: {e}")
    
    # 在轨迹末尾添加错误信息
    if action_list:
        action_list[-1]["is_error"] = is_error
        action_list[-1]["error_msg"] = error_msg
    else:
        # 如果没有任何动作，也要记录错误
        action_list.append({
            "input_seq": f"question: {question}",
            "output_seq": "初始化失败",
            "is_error": is_error,
            "error_msg": error_msg
        })
    
    # 整个轨迹结束后统一落盘
    write_a_data(action_list)

# data_path="/data/minimax-dialogue/users/ruobai/rl_r2e/data/wikiQA_debug/dev.parquet"
# data_path="/home/zhiheng/WikiRL/ragen/env/wiki/data/puzzle/dev.parquet"
data_path="/home/zhiheng/WikiRL/ragen/env/wiki/data/puzzle/test.parquet"
max_threads = 1  # 降低并发数量，避免资源过度消耗
number_to_process = 3610

def process_single_item(row):
    """处理单个数据项的包装函数"""
    question = row["extra_info"]["question"]
    gt = row["extra_info"]["selected_answer"]
    return Get_multi_turn_response(question, gt)

if __name__ == "__main__":
    data_df = pd.read_parquet(data_path)
    print(data_df.shape)
    data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 限制处理的数据量
    data_to_process = data_df.head(number_to_process)
    
    print(f"开始处理 {len(data_to_process)} 个数据项，使用 {max_threads} 个线程")
    
    # 使用线程池进行并发处理
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # 提交所有任务
        future_to_row = {
            executor.submit(process_single_item, row): idx 
            for idx, row in data_to_process.iterrows()
        }
        
        # 处理完成的任务
        completed_count = 0
        for future in as_completed(future_to_row):
            idx = future_to_row[future]
            try:
                result = future.result()  # 获取结果（如果需要的话）
                completed_count += 1
                if completed_count % 10 == 0:  # 每10个打印一次进度
                    print(f"已完成 {completed_count}/{len(data_to_process)} 个任务")
            except Exception as e:
                print(f"任务 {idx} 执行出错: {e}")
                completed_count += 1
    
    print(f"所有任务完成！总计处理了 {completed_count} 个数据项")