import json
from openai import OpenAI
import threading
import json
lock=threading.Lock()
api_key = ""
client = OpenAI(api_key = api_key,base_url= "https://api.openai.com/v1/")
with open("system_prompt.txt","r",encoding = "utf-8") as f:
    system_prompt = f.read()

user_prompt = """
Objective: {}
URL: {}
Observation: {}
HISTORY_ACTION: {}
"""

def get_response(prompt , model = "GPT-4", temperature = 0):
    #print(prompt)
    try:
        response = client.chat.completions.create(
            model = model,
            messages = [{"role": "user", "content": prompt}],
            temperature = temperature,
            max_tokens = 1024
        )
        model_answer = response.choices[0].message.content
    except:
        model_answer = "```error```"
    return model_answer

import re
def extract_command(text):
    #print(text)
    blocks = re.findall(r'```\s*([^\s].*?[^\s])\s*```', text, re.DOTALL)
    
    if not blocks:
        return None

    last_command = blocks[-1].strip()
    last_command = last_command.replace("```","")
    return last_command.strip()

def write_a_data(input,output):
    written_data = {"input_seq":input,"output_seq":output}
    lock.acquire()
    with open("gene_data.json","a",encoding = "utf-8") as fw:
        fw.write(json.dumps(written_data,ensure_ascii=False) + "\n")
    lock.release()

def Get_multi_turn_response(obj, url, obs):
    history = "\n"
    for i in range(5):
        real_prompt = user_prompt.format(obj, url, obs, history)
        prompt = system_prompt + "\n\n" + real_prompt
        response = get_response(prompt)
        last_command = extract_command(response)
        history = history + last_command + "\n"
        
        write_a_data(prompt,response)
        if "stop" in last_command:
            return

data_path = "./test.txt"
max_threads = 64

if __name__ == "__main__":
    with open(data_path, "r" ,encoding = "utf-8") as f:
        lines = f.readlines()
    cnt = 0
    threads = []
    for line in lines:
        obj, url, obs = line.split('\t')
        t = threading.Thread(target=Get_multi_turn_response , args = (obj, url, obs))
        threads.append(t)
        cnt += 1
        if cnt % max_threads == 0 or cnt == len(lines):
            for t in threads:
                t.start()
                t.join()
            threads = []