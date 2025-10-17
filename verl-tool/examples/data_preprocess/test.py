import pandas as pd
import re
import jsonlines
import os
import asyncio
import time
from collections import deque
from openai import OpenAI

# 读取系统 prompt
with open("sys_eval_prompt.txt","r",encoding="utf-8") as f:
    eval_prompt = f.read()

# 设置环境变量
os.environ['OPENAI_API_KEY'] = "yAWch3FwxYNiiNF4aSd9YiM5@3892"

client = OpenAI(base_url="http://v2.open.venus.oa.com/llmproxy")

# ----------------- 限速器 -----------------
class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls   # 最大调用次数
        self.period = period         # 时间窗口（秒）
        self.calls = deque()         # 保存调用时间戳
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            # 移除窗口外的调用
            while self.calls and self.calls[0] <= now - self.period:
                self.calls.popleft()
            if len(self.calls) >= self.max_calls:
                # 需要等待
                sleep_time = self.period - (now - self.calls[0])
                await asyncio.sleep(sleep_time)
            self.calls.append(time.time())

# 每分钟最多 50 次
rate_limiter = RateLimiter(50, 60)

# ----------------- 请求函数 -----------------
def get_response(prompt , model = "gpt-4.1", temperature = 0):
    response = client.chat.completions.create(
        model = model,
        messages = [{"role": "user", "content": prompt}],
        temperature = temperature,
        max_tokens = 1024
    )
    return response.choices[0].message.content

# 异步封装
async def same(question, gt, ans, sem, save_writer):
    prompt = eval_prompt.format(question, gt, ans)

    async def run(model):
        async with sem:  # 限制并发
            await rate_limiter.acquire()  # 限制速率
            return await asyncio.to_thread(get_response, prompt, model)

    answer_gpt, answer_gemini, answer_claude = await asyncio.gather(
        run("gpt-4.1"),
        run("gemini-2.5-flash"),
        run("claude-3-7-sonnet-20250219"),
    )

    print(answer_claude, answer_gemini, answer_gpt)

    # 保存结果
    save_writer.write({
        "question": question,
        "ground_truth": gt,
        "answer": ans,
        "gpt-4.1": answer_gpt,
        "gemini-2.5-flash": answer_gemini,
        "claude-3-7-sonnet-20250219": answer_claude
    })

    # 修改判断逻辑：只要有 2 个 "yes" 就算正确
    yes_count = sum("yes" in x.lower() for x in [answer_claude, answer_gemini, answer_gpt])
    if yes_count >= 2:
        return 1
    return 0

# ---------------- 主逻辑 ----------------
async def main():
    data_path = "test.parquet"
    data_df = pd.read_parquet(data_path)

    gt_answer = {
        row["extra_info"]["question"]: row["extra_info"]["selected_answer"]
        for _, row in data_df.iterrows()
    }

    gen_file = 'nq_main.jsonl'
    with jsonlines.open(gen_file) as reader:   
        gen_data = list(reader)

    steps = 0
    suc = 0
    emp = 0

    sem = asyncio.Semaphore(10)  # 限制同时最多 10 并发

    # 打开结果保存文件
    with jsonlines.open("results.jsonl", mode="w") as save_writer:

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