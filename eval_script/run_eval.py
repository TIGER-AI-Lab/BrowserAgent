# from ragen.env.wiki.env import WikiQAEnv
import pandas as pd
import os
import argparse
import time
import re
import json
import requests
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
# from openai_api import call_gpt4o_single
from model_api import call_sglang_single, call_gemini_single
from sglang_api import call_sglang_local, init_sglang_local

def traj2str(traj, obs_type = 'single'): # single, full or tuncated
    if obs_type == 'single':
        traj = [traj[0], traj[-1]]
    ans = ""
    for item in traj:
        if item["role"] == "system":
            ans += item["observation"]
        elif item["role"] == "user":
            ans += item["observation"]
        elif item["role"] == "assistant":
            # if item observation not start with <|im_start|>assistant, add it
            obs = item["observation"]
            if not obs.startswith("<|im_start|>assistant"):
                obs = "<|im_start|>assistant" + obs
            # if obs not end with <|im_end|>, add it
            if not obs.endswith("<|im_end|>"):
                obs = obs + "<|im_end|>"
            ans += obs
        else:
            raise ValueError("role not recognized")
    ans = ans + "<|im_start|>assistant"
    from prompt_process import tuncate_input, tuncate_plain, direct_plain
    # if obs_type == "tuncate_input":
    #     print(f"obs_type: {obs_type}")
    #     print(f"before: {ans}")
    if obs_type == 'tuncate_input':
        ans = tuncate_input(ans)
    elif obs_type == 'tuncate_plain':
        ans = tuncate_plain(ans)
    elif obs_type == 'direct_plain':
        ans = direct_plain(ans)
    elif obs_type != 'full' and obs_type != 'single':
        raise ValueError("obs_type not recognized")
    # if obs_type == "tuncate_input":
    #     print(f"after: {ans}")
    #     exit(1)
    return ans

def get_result(data_id, question, answer, init_url, model_port = 5001, server_port = 5002, max_steps = 10,
            curr_obs_type = "tunc_plain", model_api = "sglang"):
    browser_server_url = f"http://localhost:{server_port}/get_observation"
    def extract_stop_content(input_str: str) -> str:
        match = re.search(r"```stop\s*\[([^\]]*)\]", input_str)
        if match:
            return match.group(1)
        return ""
    import uuid
    trajectory_id = str(uuid.uuid4())
    from mini_webarena.create_dataset import TEMPLATES, WIKI_LANDING
    obs_traj = [
        {"role": "system", "observation": TEMPLATES['qwen-instruct']['system']},
    ]
    payload = {
        "trajectory_ids": [trajectory_id],
        "actions": [""],
        "extra_fields": [{"question": question, "gt": answer, "url": init_url}]
    }
    response = requests.post(browser_server_url, json=payload, timeout=300).json()
    new_render, done, valid = response["observations"][0], response["dones"][0], response["valids"][0]
    obs_traj.append({"role": "user", "observation": new_render})

    trajectory = []
    pred, score, answer_score = "", 0, 0
    from mini_webarena.rl_utils import format_score
    from mini_webarena.evaluator import fuzzy_match
    for step_num in range(10):
        observation, curr_observation = traj2str(obs_traj, obs_type='single'), traj2str(obs_traj, obs_type=curr_obs_type)
        if model_api == "sglang":
            action = call_sglang_local(curr_observation)
        elif model_api == "gemini":
            action = call_gemini_single(curr_observation, port = model_port)
        else:
            raise ValueError("model_api not recognized")
        trajectory.append((data_id, observation, curr_observation, action))
        obs_traj.append({"role": "assistant", "observation": action})
        payload["actions"] = [action]
        response = requests.post(browser_server_url, json=payload, timeout=300).json()
        new_render, done, valid = response["observations"][0], response["dones"][0], response["valids"][0]
        if not valid:
            pass
        obs_traj.append({"role": "user", "observation": new_render})
        score += format_score(action, valid)
        if done:
            pred = extract_stop_content(action)
            answer_score = fuzzy_match(pred, answer)
        if step_num >= max_steps or done:
            print(f"Breaking at step {step_num} with done={done} or max_steps={max_steps}")
            break
    return trajectory, question, answer, pred, score, answer_score, data_id


def get_result_wrapper(args):
    """
    Wrapper to unpack arguments for `get_result`.
    """
    return get_result(*args)


def generate_annotation(
        data_path="/home/zhiheng/WikiRL/ragen/env/wiki/data/puzzle/train.parquet",
        num_samples=100,
        traj_csv_path="traj_0.5b.csv",
        result_csv_path="result_0.5b.csv",
        init_url="http://tigerai.ca/wiki/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
        model_port=5001,
        server_port=5002,
        max_steps=10,
        curr_obs_type="direct_plain",
        model_api="sglang"
):
    data_df = pd.read_parquet(data_path)

    if os.path.exists(traj_csv_path):
        traj_df = pd.read_csv(traj_csv_path)
    else:
        traj_df = pd.DataFrame(columns=["data_id", "single_obs", "full_obs", "output"])

    if os.path.exists(result_csv_path):
        result_df = pd.read_csv(result_csv_path)
    else:
        result_df = pd.DataFrame(columns=["data_id", "input", "gt", "pred", "score", "answer_score"])

    # 立刻过滤掉input为空的行，并写回csv，保证异常样本可以重新推理
    if not result_df.empty:
        result_df = result_df[result_df["input"] != ""]
        result_df.to_csv(result_csv_path, index=False)

    processed_ids = set(result_df["data_id"].unique())
    task_args = []

    count = 0
    for i, row in data_df.iterrows():
        if count >= num_samples:
            break
        data_id = row["id"]
        if data_id in processed_ids:
            count += 1
            continue
        question = row["question"]
        gt = row["golden_answers"][0]
        task_args.append((data_id, question, gt, init_url, model_port, server_port, max_steps, curr_obs_type, model_api))
        count += 1

    with ProcessPoolExecutor(max_workers=16) as executor:
        future_to_args = {
            executor.submit(get_result_wrapper, args): args[0]
            for args in task_args
        }
        for future in tqdm(as_completed(future_to_args), total=len(task_args)):
            data_id = future_to_args[future]
            try:
                result = future.result(timeout=300)
                trajectory, question, gt, pred, score, answer_score, data_id = result
                for (data_id, single_obs, full_obs, output) in trajectory:
                    traj_df.loc[len(traj_df)] = [data_id, single_obs, full_obs, output]
                result_df.loc[len(result_df)] = {
                    "data_id": data_id,
                    "input": question,
                    "gt": gt,
                    "pred": pred,
                    "score": score,
                    "answer_score": answer_score
                }
            except Exception as e:
                print(f"[Error] data_id={data_id} failed with error: {e}")
                result_df.loc[len(result_df)] = {
                    "data_id": data_id,
                    "input": "",
                    "gt": "",
                    "pred": "",
                    "score": 0,
                    "answer_score": 0
                }
                continue
            traj_df.to_csv(traj_csv_path, index=False)
            result_df.to_csv(result_csv_path, index=False)

    print(f"Done processing. Successfully processed {len(task_args)} new samples.")


def test_generate_annotation(
        data_path="/home/zhiheng/WikiRL/ragen/env/wiki/data/puzzle/train.parquet",
        num_samples=100,
        traj_csv_path="traj_0.5b.csv",
        result_csv_path="result_0.5b.csv",
        init_url="http://tigerai.ca/wiki/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
        model_port=5001,
        server_port=5002,
        max_steps=10,
        curr_obs_type="direct_plain",
        model_api="sglang"
):
    import sys
    data_df = pd.read_parquet(data_path)

    if os.path.exists(traj_csv_path):
        traj_df = pd.read_csv(traj_csv_path)
    else:
        traj_df = pd.DataFrame(columns=["data_id", "single_obs", "full_obs", "output"])

    if os.path.exists(result_csv_path):
        result_df = pd.read_csv(result_csv_path)
    else:
        result_df = pd.DataFrame(columns=["data_id", "input", "gt", "pred", "score", "answer_score"])

    processed_ids = set(result_df["data_id"].unique())
    count = 0
    for i, row in data_df.iterrows():
        if count >= num_samples:
            break
        data_id = row["id"]
        if data_id in processed_ids:
            count += 1
            continue
        question = row["question"]
        gt = row["golden_answers"][0]
        try:
            result = get_result(data_id, question, gt, init_url, model_port, server_port, max_steps, curr_obs_type, model_api)
            trajectory, question, gt, pred, score, answer_score, data_id = result
            for (data_id, single_obs, full_obs, output) in trajectory:
                traj_df.loc[len(traj_df)] = [data_id, single_obs, full_obs, output]
            result_df.loc[len(result_df)] = {
                "data_id": data_id,
                "input": question,
                "gt": gt,
                "pred": pred,
                "score": score,
                "answer_score": answer_score
            }
        except Exception as e:
            raise e
            print(f"[Error] data_id={data_id} failed with error: {e}")
            result_df.loc[len(result_df)] = {
                "data_id": data_id,
                "input": "",
                "gt": "",
                "pred": "",
                "score": 0,
                "answer_score": 0
            }
        traj_df.to_csv(traj_csv_path, index=False)
        result_df.to_csv(result_csv_path, index=False)
        print(f"Test模式下已处理第一个样本，退出。")
        sys.exit(1)
    print(f"Test模式下未处理任何新样本。")


def timed_run(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"✅ Finished '{func.__name__}' in {elapsed:.2f} seconds.")
    return result


# -----------------------------------------------------------------------------
# Main with argparse
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_port", type=int, default=5001, help="Port used by the language model")
    parser.add_argument("--server_port", type=int, default=5002, help="Port used by the environment server")
    parser.add_argument("--num_samples", type=int, default=300, help="Number of samples to process")
    parser.add_argument("--max_steps", type=int, default=10, help="Max steps to use for each sample")
    parser.add_argument("--data_path", type=str,
                        default="./train.parquet",
                        help="Path to the puzzle data file")
    parser.add_argument("--traj_csv_path", type=str, default="train_traj_1.5b.csv",
                        help="Path to save trajectory CSV")
    parser.add_argument("--result_csv_path", type=str, default="train_result_1.5b.csv",
                        help="Path to save results CSV")
    parser.add_argument("--init_url", type=str,
                        default="http://tigerai.ca/wiki/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
                        help="Initial URL")
    parser.add_argument("--obs_type", type=str, default="direct_plain",
                        help="Observation type: single, full, tuncate_input, direct_plain, tuncate_plain")
    parser.add_argument("--model_api", type=str, default="sglang",
                        help="Model api: sglang, gemini") # TODO, need add more
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B",
                        help="Your sglang model path")
    parser.add_argument("--cuda", type=str, default="0",
                        help="If you use sglang, please specify your cuda id")
    parser.add_argument("--test_mode", action="store_true", default=False, help="Test模式，顺序执行第一个样本后退出")
    return parser.parse_args()

import pandas as pd
from tqdm import trange
from mini_webarena.evaluator import metric_exact_match, metric_heuristic

def eval_inference(data_path: str,
                   result_path: str,
                   save_path: str) -> None:
    """
    Evaluate prediction results against gold answers by aligning on `data_id`.

    Args
    ----
    data_path   : str  – Parquet file containing columns ['id', 'question', 'golden_answers'].
    result_path : str  – CSV file containing model outputs with column `data_id`.
    save_path   : str  – Where to write the evaluation CSV.

    Notes
    -----
    *  Assumes `result_path` has a column called `data_id` that matches `id` in `data_path`.
    *  Adds four columns to the result CSV: `gts`, `em`, `heur`, and `aligned`.
    """

    # ---------- 1. Load data ----------
    data_df   = pd.read_parquet(data_path)
    result_df = pd.read_csv(result_path)

    # ---------- 2. Align by ID ----------
    # Rename for clarity and merge; keep only one copy of the key.
    data_df   = data_df.rename(columns={'id': 'data_id'})
    merged_df = result_df.merge(
        data_df[['data_id', 'golden_answers']],
        on='data_id',
        how='left',
        validate='many_to_one'          # raise if duplicate gold rows
    )

    # Mark rows that failed to align (should be 0 in ideal case).
    merged_df['aligned'] = merged_df['golden_answers'].notna()

    if not merged_df['aligned'].all():
        missing = merged_df[~merged_df['aligned']]
        print(f"[WARN] {len(missing)} predictions had no matching gold answers!")

    # ---------- 3. Metric computation ----------
    em_scores, heur_scores = [], []

    for i in trange(len(merged_df), desc="Scoring"):
        gold = merged_df.at[i, 'golden_answers']
        pred = str(merged_df.at[i, 'pred'])
        em_scores.append(metric_exact_match(gold, pred))
        heur_scores.append(metric_heuristic(gold, pred))

    # Attach results
    merged_df['gts']  = merged_df['golden_answers']
    merged_df['em']   = em_scores
    merged_df['heur'] = heur_scores

    # ---------- 4. Aggregate & report ----------
    total   = len(merged_df)
    em_avg  = sum(em_scores)   / total if total else 0
    heur_avg= sum(heur_scores) / total if total else 0

    print(f"Total samples: {total}")
    print(f"Exact Match Score (em): {em_avg:.6f}")
    print(f"Heuristic Score (heur): {heur_avg:.6f}")

    # ---------- 5. Persist ----------
    merged_df.to_csv(save_path, index=False)


def main():
    args = parse_args()
    if getattr(args, 'test_mode', False):
        if args.model_api == "sglang":
            init_sglang_local(args.model_path, args.cuda)
        test_generate_annotation(
            data_path=args.data_path,
            num_samples=args.num_samples,
            traj_csv_path=args.traj_csv_path,
            result_csv_path=args.result_csv_path,
            init_url=args.init_url,
            model_port=args.model_port,
            server_port=args.server_port,
            max_steps=args.max_steps,
            curr_obs_type=args.obs_type,
            model_api=args.model_api)
        return
    data_len = len(pd.read_parquet(args.data_path))
    result_len = 0
    if os.path.exists(args.result_csv_path):
        result_len = len(pd.read_csv(args.result_csv_path))
    if result_len == data_len:
        print(f"结果已存在且完整（{result_len}条），直接评测...")
        eval_inference(
            data_path=args.data_path,
            result_path=args.result_csv_path,
            save_path=args.result_csv_path.replace('.csv', '_eval.csv')
        )
        return
    if args.model_api == "sglang":
        init_sglang_local(args.model_path, args.cuda)
    timed_run(
        generate_annotation,
        data_path=args.data_path,
        num_samples=args.num_samples,
        traj_csv_path=args.traj_csv_path,
        result_csv_path=args.result_csv_path,
        init_url=args.init_url,
        model_port=args.model_port,
        server_port=args.server_port,
        max_steps=args.max_steps,
        curr_obs_type=args.obs_type,
        model_api=args.model_api
    )
    eval_inference(
        data_path=args.data_path,
        result_path=args.result_csv_path,
        save_path=args.result_csv_path.replace('.csv', '_eval.csv')
    )


if __name__ == "__main__":
    main()