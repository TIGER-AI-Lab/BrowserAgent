import pandas as pd
import concurrent.futures
from openai import OpenAI
import os
import tqdm
import numpy as np
import argparse


# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
# 保持build_messages和infer_one函数主体不变，但infer_one增加client和model_name参数

def build_messages(row):
    # print(row)
    prompt = row['prompt']
    if isinstance(prompt, str):
        import ast
        prompt = ast.literal_eval(prompt)
    return prompt

def infer_one(row, client, model_name):
    messages = build_messages(row)
    print(messages)
    import sys
    sys.exit()
    extra_body = row['extra_info']
    if isinstance(extra_body, str):
        import ast
        extra_body = ast.literal_eval(extra_body)
    if hasattr(extra_body, 'item'):
        extra_body = extra_body.item()
    def convert_ndarray(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_ndarray(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_ndarray(x) for x in obj]
        else:
            return obj
    extra_body = convert_ndarray(extra_body)
    # 新增：从extra_body里取question和golden_answers
    if isinstance(extra_body, dict):
        question = extra_body.get("question", row.get("question", None))
        golden_answers = extra_body.get("golden_answers", row.get("golden_answers", None))
    else:
        question = row.get("question", None)
        golden_answers = row.get("golden_answers", None)
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=10240,
            top_p=1,
            n=1,
            extra_body=extra_body
        )
        content = completion.choices[0].message.content
        finish_reason = completion.choices[0].finish_reason
    except Exception as e:
        raise e
        content = f"[ERROR]{str(e)}"
        finish_reason = "error"
    return {
        "id": row.get("id", None),
        "question": question,
        "golden_answers": golden_answers,
        "output": content,
        "finish_reason": finish_reason
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/home/zhiheng/WikiRL/ragen/env/wiki/data/puzzle/dev.parquet")
    # parser.add_argument('--result_dir', type=str, default="/data/minimax-dialogue/ruobai/cogito/verl-tool/eval_service/result_new")
    parser.add_argument('--result_dir', type=str, default="/home/yutao/verl-tool/eval_service/result")
    parser.add_argument('--model_name', type=str, default="/home/yutao/model/Qwen2.5-3B-Instruct")
    parser.add_argument('--api_port', type=int, default=5001)
    parser.add_argument('--api_key', type=str, default="sk-proj-1234567890")
    parser.add_argument('--max_workers', type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, f"{os.path.basename(args.model_name)}_result.csv")
    client = OpenAI(api_key=args.api_key, base_url=f"http://localhost:{args.api_port}")

    df = pd.read_parquet(args.data_path)
    if 'id' not in df.columns:
        df['id'] = range(len(df)) 
    # print(df.head)
    # print(df.iloc[0]['extra_info'])
    # exit(1)

    # 尝试加载已存在的csv，获取已完成id
    if os.path.exists(result_path):
        result_df = pd.read_csv(result_path)
        finished_ids = set(result_df['id'].dropna().unique())
        print(f"已完成推理样本数: {len(finished_ids)}")
    else:
        result_df = pd.DataFrame(columns=["id", "question", "golden_answers", "output", "finish_reason"])
        finished_ids = set()

    # 过滤未完成的样本
    todo_rows = [row for idx, row in df.iterrows() if row['id'] not in finished_ids]
    print(f"待推理样本数: {len(todo_rows)}")

    # 推理并实时append到csv
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(infer_one, row, client, args.model_name): row['id'] for row in todo_rows}
        for f in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="推理进度"):
            # print("Fetching result...")
            # try:
            #     result = f.result(timeout=300)  # 设置超时时间
            # except concurrent.futures.TimeoutError:
            #     print("Timeout on a future")
            #     continue
            # except Exception as e:
            #     print(f"Exception occurred: {e}")
            #     continue
            
            try:
                result = f.result()
            except Exception as e:
                raise e
                exit(1)
                # 异常样本写入空output
                row_id = futures[f]
                result = {"id": row_id, "question": "", "golden_answers": "", "output": f"[ERROR]{str(e)}", "finish_reason": "error"}
            # 追加到csv
            result_df.loc[len(result_df)] = result
            result_df.to_csv(result_path, index=False)
    print(f"推理完成，结果已保存到 {result_path}")

if __name__ == "__main__":
    main()