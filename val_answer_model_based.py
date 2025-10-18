import pandas as pd
import re
import jsonlines
import os
import asyncio
import time
from collections import deque
from openai import OpenAI
import argparse

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


os.environ['OPENAI_API_KEY'] = ""

client = OpenAI(base_url="https://api.openai.com/v1/")


class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls  
        self.period = period        
        self.calls = deque()        
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            
            while self.calls and self.calls[0] <= now - self.period:
                self.calls.popleft()
            if len(self.calls) >= self.max_calls:
               
                sleep_time = self.period - (now - self.calls[0])
                await asyncio.sleep(sleep_time)
            self.calls.append(time.time())


rate_limiter = RateLimiter(50, 60)


def get_response(prompt , model = "gpt-4.1", temperature = 0):
    response = client.chat.completions.create(
        model = model,
        messages = [{"role": "user", "content": prompt}],
        temperature = temperature,
        max_tokens = 1024
    )
    return response.choices[0].message.content


async def same(question, gt, ans, sem, save_writer):
    prompt = eval_prompt.format(question, gt, ans)

    async def run(model):
        async with sem:  
            await rate_limiter.acquire()  
            return await asyncio.to_thread(get_response, prompt, model)

    answer_gpt, answer_gemini, answer_claude = await asyncio.gather(
        run("gpt-4.1"),
        run("gemini-2.5-flash"),
        run("claude-3-7-sonnet-20250219"),
    )

    print(answer_claude, answer_gemini, answer_gpt)

  
    save_writer.write({
        "question": question,
        "ground_truth": gt,
        "answer": ans,
        "gpt-4.1": answer_gpt,
        "gemini-2.5-flash": answer_gemini,
        "claude-3-7-sonnet-20250219": answer_claude
    })

    
    yes_count = sum("yes" in x.lower() for x in [answer_claude, answer_gemini, answer_gpt])
    if yes_count >= 2:
        return 1
    return 0


async def main():
    data_df = pd.read_parquet(args.data_path)

    gt_answer = {
        row["extra_info"]["question"]: row["extra_info"]["selected_answer"]
        for _, row in data_df.iterrows()
    }


    with jsonlines.open(args.gen_file) as reader:   
        gen_data = list(reader)

    steps = 0
    suc = 0
    emp = 0

    sem = asyncio.Semaphore(10)

  
    with jsonlines.open(args.output_file, mode="w") as save_writer:

        async def process(data):
            nonlocal suc, steps, emp
            content = data['trajectory']
            input_seq = content[-1]['input_seq']
            output_seq = content[-1]['output_seq']

            question = re.findall(r'Objective: (.*?)\nObservation', input_seq)[0]

            if not re.findall(r"```(.*?)```", output_seq):
                answer = " "
            else:
                answer = re.findall(r"```(.*?)```", output_seq)[0]

            if 'stop' in answer:
                try:
                    ans = re.findall(r"\[(.*?)\]", answer)[0]
                except:
                    ans = ""
                ground_truth = gt_answer[question]
                try:
                    if await same(question, ground_truth, ans, sem, save_writer):
                        suc += 1
                        steps += data['trajectory_length']
                except Exception as e:
                    print(f"错误: {e}")
            else:
                emp += 1

        tasks = [process(data) for data in gen_data]
        await asyncio.gather(*tasks)

    print(f"问题数目：{len(gen_data)}")
    print(f"回答正确数目：{suc}")
    print(f"未回答数目：{emp}")
    print(f"平均步数：{steps/suc if suc > 0 else 0:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
