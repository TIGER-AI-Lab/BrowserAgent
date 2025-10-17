# from ragen.env.wiki.env import WikiQAEnv
import pandas as pd
import os
import argparse
import time
import re
import json
import requests
from concurrent.futures import ProcessPoolExecutor
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
    from sft_data_process import tuncate_input, tuncate_plain, direct_plain
    # print(f"obs_type: {obs_type}")
    if obs_type == 'tuncate_input':
        ans = tuncate_input(ans)
    elif obs_type == 'tuncate_plain':
        ans = tuncate_plain(ans)
    elif obs_type == 'direct_plain':
        ans = direct_plain(ans)
    elif obs_type != 'full' and obs_type != 'single':
        raise ValueError("obs_type not recognized")
    return ans

def get_result(data_id, question, answer, init_url, model_port = 5001, server_port = 5002, max_steps = 10,
            #curr_obs_type = "direct_plain"):
            curr_obs_type = "tunc_plain", model_api = "sglang"):
    # first random generate a uuid
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
        # {"role": "user", "observation": WIKI_LANDING.format(objective=question)},
    ]
    # print(question)
    # print("#"*100)
    # print(obs_traj[0]['observation'])
    # print(obs_traj[1]['observation'])
    # exit(1)
    payload = {
        "trajectory_ids": [trajectory_id],
        "actions": [""],
        "extra_fields": [{"question": question, "gt": answer, "url": init_url}]
    }
    response = requests.post(browser_server_url, json=payload).json() # Get first obs
    new_render, done, valid = response["observations"][0], response["dones"][0], response["valids"][0]
    # print(f"new_render: {new_render}")
    # print("="*100)
    obs_traj.append({"role": "user", "observation": new_render})

    trajectory = []
    pred, score, answer_score = "", 0, 0
    from mini_webarena.rl_utils import format_score
    from mini_webarena.evaluator import fuzzy_match
    for step_num in range(10):
        observation, curr_observation = traj2str(obs_traj, obs_type='single'), traj2str(obs_traj, obs_type=curr_obs_type)
        if model_api == "sglang":
            # action = call_sglang_single(curr_observation, port = model_port)  # to simplify the problem
            action = call_sglang_local(curr_observation)  # to simplify the problem
        elif model_api == "gemini":
            action = call_gemini_single(curr_observation, port = model_port)
        else:
            raise ValueError("model_api not recognized")
        # print(curr_observation)
        # print(f"#" * 100)
        trajectory.append((data_id, observation, curr_observation, action))
        obs_traj.append({"role": "assistant", "observation": action})
        payload["actions"] = [action]
        # print(payload)
        # print(f"#" * 100)
        # try:
        if True:
            response = requests.post(browser_server_url, json=payload).json()
            # print(response)
            new_render, done, valid = response["observations"][0], response["dones"][0], response["valids"][0]
            # print(f"new_render: {new_render}")
            # print("="*100)
            if not valid:
                print(f"Invalid payload: {payload}")
                print("#"* 100)
                print(f"Invalid new_render: {new_render}")
                # print("#"* 100)
                # print(f"Invalid done: {done}")
                # print("#"* 100)
                # print(f"Invalid valid: {valid}")
                # print("#"* 100)
                # print(f"Invalid step_num: {step_num}")
                # exit(1)
            obs_traj.append({"role": "user", "observation": new_render})
            score += format_score(action, valid)
            # print(new_render)
            # print(f"step {step_num}, action: {action}, observation: {observation}, done: {done}, valid: {valid}")
            # print(f"#"*100)
            # print(f"#" * 100)
            # exit(1)
            if done:
                pred = extract_stop_content(action)
                answer_score = fuzzy_match(pred, answer)
        # except Exception as e:
        #     score += format_score(action, False)
        #     print(f"step {step_num}, error: {e}")
        if step_num >= max_steps or done:
            print(f"Breaking at step {step_num} with done={done} or max_steps={max_steps}")
            break
    # TODO: pred, score, answer_score
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
        model_api="sglang" # TODO, current is sglang or gemini
):
    """
    Main function that:
    1) Reads data from `data_path`
    2) Processes up to `num_samples` items
    3) Uses concurrent calls to `get_result`
    4) Logs the results into CSV files (trajectories and final results).
    """
    data_df = pd.read_parquet(data_path)

    # load or create trajectory CSV
    if os.path.exists(traj_csv_path):
        traj_df = pd.read_csv(traj_csv_path)
    else:
        traj_df = pd.DataFrame(columns=["data_id", "single_obs", "full_obs", "output"])

    # load or create result CSV
    if os.path.exists(result_csv_path):
        result_df = pd.read_csv(result_csv_path)
    else:
        result_df = pd.DataFrame(columns=["data_id", "input", "gt", "pred", "score", "answer_score"])

    processed_ids = set(result_df["data_id"].unique())
    task_args = []

    count = 0
    for i, row in data_df.iterrows():
        if count >= num_samples:
            break
        # print("row", row) # "id", "question", "golden_answers"

        data_id = row["id"]
        if data_id in processed_ids:
            count += 1
            continue
        question = row["question"]
        gt = row["golden_answers"][0]

        # Notice we now also pass model_port, server_port, max_steps here:
        task_args.append((data_id, question, gt, init_url, model_port, server_port, max_steps, curr_obs_type, model_api))
        count += 1

    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = executor.map(get_result_wrapper, task_args)
        for trajectory, question, gt, pred, score, answer_score, data_id in tqdm(futures, total=len(task_args)):
            # update trajectory CSV
            for (data_id, single_obs, full_obs, output) in trajectory:
                traj_df.loc[len(traj_df)] = [data_id, single_obs, full_obs, output]
            # update results CSV
            result_df.loc[len(result_df)] = {
                "data_id": data_id,
                "input": question,
                "gt": gt,
                "pred": pred,
                "score": score,
                "answer_score": answer_score
            }
            # save partial progress
            traj_df.to_csv(traj_csv_path, index=False)
            result_df.to_csv(result_csv_path, index=False)
            # exit(1)

    print(f"Done processing. Successfully processed {len(task_args)} new samples.")


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
                        help="Observation type: single, full, tuncated, direct_plain, tunc_plain")
    parser.add_argument("--model_api", type=str, default="sglang",
                        help="Model api: sglang, gemini") # TODO, need add more
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B",
                        help="Your sglang model path")
    parser.add_argument("--cuda", type=str, default="0",
                        help="If you use sglang, please specify your cuda id")
    return parser.parse_args()


def main():
    args = parse_args()
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


if __name__ == "__main__":
    main()
"""
# Step 1: Start browser server; under ~/verl-tool$; Need change the port in line 37; 
bash examples/server/wikiRL_server.sh

# Step 2: Start sglang server; under ~/mini_webarena$; Need change the port in hf_api.py
python -m mini_webarena.server_sglang


# Step 3: Run the script
python concurrent_generate.py
"""