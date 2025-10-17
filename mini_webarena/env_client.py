# import nest_asyncio
# nest_asyncio.apply()

import time
import uuid
import requests
import asyncio
import re
import random
from copy import deepcopy
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

# ------------------ 这里是client的HTTP请求封装，可直接使用 ------------------
def client_start(server_url, questions, gts, keys):
    """
    对应服务端的 /start
    将需要的 questions, gts, keys 组装进请求中
    """
    print("################# client_start ####################")
    print(server_url, questions, gts, keys)
    payload = {
        "questions": questions,
        "gts": gts,
        "keys": keys
    }
    print(payload)
    try:
        resp = requests.post(f"{server_url}/start", json=payload)
    except requests.exceptions.HTTPError as e:
        print("HTTPError:", e)
        print("Response content:", resp.text)
        raise
    resp.raise_for_status()

    print("################# client_start ####################")
    return resp.json()  # 服务端返回的 dict

def client_step(server_url, query, keys):
    """
    对应服务端的 /step
    query 是个列表，应与 keys 等长（一一对应）
    但此处只演示单 key 用法，所以 query 只有 1 个元素
    """
    print("################# client_step ####################")
    print(server_url, query, keys)
    payload = {
        "query": query,
        "keys": keys
    }
    resp = requests.post(f"{server_url}/step", json=payload)
    resp.raise_for_status()
    print(resp.json())
    print("################# client_step ####################")
    return resp.json()  # 服务端返回 dict
# --------------------------------------------------------------


# ================ 下面是你原先的基础类和一些依赖 ================
# 假设你的 BaseLanguageBasedEnv 就在同目录下的 base.py
from .env_base import BaseLanguageBasedEnv

# 这里的 ActionTypes、create_xxx_action 等，是你原先用来解析模型输出动作的
from .browser_actions import (
    ActionTypes,
    create_stop_action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
)

# 用于计算 reward 的辅助函数
from .rl_utils import format_score
from .evaluator import fuzzy_match

# 用于 prompt 拼接的模板
from .create_dataset import TEMPLATES

# 假设这是你用于解析大模型输出中动作部分的工具
from .agent import construct_promptConstructor

# =============== 以下是重写后的 WikiQAEnv ===============
class WikiQAEnv(BaseLanguageBasedEnv):
    def __init__(
        self,
        dataset=None,     # 包含 (question, answer) 等信息的 DataFrame
        seed: int = 0,    # 哪一行数据
        max_steps: int = 10,
        threshold: float = 0.7,
        prompt_format: str = "single",  # full, single 等等
        server_url: str = "http://localhost:8004",
    ):
        """
        说明:
         - 该环境不再使用 ScriptBrowserEnv 或 AsyncScriptBrowserEnv;
           而是通过HTTP请求与远端的服务交互 (server_url 对应服务端地址).
         - 每个环境实例通过一个独立的 UUID 来区分, 并在 /start 时只启动一个索引.

        Args:
            dataset: 你的数据集 (DataFrame)，至少需要含 'extra_info' 字段，
                     其中 'extra_info' 又需包含 'question' 和 'selected_answer'.
            seed: 选取 dataset 中第几行作为本轮的 QA 问题.
            max_steps: 最大交互步数，超出后 env.done = True.
            threshold: final answer 与 ground truth 的匹配度 >= threshold 则视为成功.
            prompt_format: 指定渲染 observation 的方式，常见为 "full" 或 "single".
            server_url: 远端服务地址 (默认 http://localhost:8004).
        """
        super().__init__()
        self.dataset = dataset
        self.seed = seed
        self.max_steps = max_steps
        self.threshold = threshold
        self.prompt_format = prompt_format
        self.server_url = server_url

        # 当前 step, 是否结束, 累计 reward
        self.current_step = 0
        self.done = False
        self.reward = 0.0

        # 从 dataset 中取出 question / ground_truth
        datapoint = self.dataset.iloc[self.seed]
        self.question = datapoint["extra_info"]["question"]
        self.gt = datapoint["extra_info"]["selected_answer"]

        print("=" * 30)
        print("WikiQAEnv init (Client Mode)")
        print(f"Question: {self.question}")
        print(f"Ground Truth: {self.gt}")
        print("=" * 30)

        self.pred = None
        self.answer_similarity = 0.0
        self.answer_made = False  # 是否已经做出 CHECK/STOP 动作
        self.obs_modality = "text"

        # 准备 prompt 模板
        self.template_dict = TEMPLATES['qwen-instruct']

        # 用于解析大模型输出动作
        self.prompt_constructor, self.tokenizer, _ = construct_promptConstructor(
            "Qwen/Qwen2.5-14B-Instruct", None
        )

        # 用于给服务端启动浏览器时的初始 index
        # 比如给每个 env 分配一个独立的 index = 100 + seed (也可自行定义)
        self.index = self.seed
        # 给自己分配一个独立的 UUID
        self.key = str(uuid.uuid4())

        # server 可能需要的起始 URL，你也可以改成自己需要的
        self.url = "http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"

        # 初始化 history
        # 先不 /start，这里只占位。等真正 reset 时再调 /start

        self.history = [{"role": "system"}]

        # 是否需要追踪其他变量
        self._reset_tracking_variables()

    def __str__(self):
        return f"WikiQAEnv(seed={self.seed}, question={self.question}, answer={self.gt}, key={self.key})"

    def reset(self, seed: Optional[int] = None) -> str:
        """
        reset 环境, 并调用 server 端的 /start 启动对应 index 的浏览器 (仅 1 个).
        返回初始 observation（可供调用者打印或存储）。
        """
        if seed is not None:
            self.seed = seed

        # 重置内部状态
        self.current_step = 0
        self.done = False
        self.reward = 0.0

        # 重新选取数据
        datapoint = self.dataset.iloc[self.seed]
        self.question = datapoint["extra_info"]["question"]
        self.gt = datapoint["extra_info"]["selected_answer"]
        self.pred = None
        self.answer_similarity = 0.0
        self.answer_made = False

        # 生成新的 index、key (你也可以选择只在 __init__ 阶段生成 key；此处看需求)
        self.index = self.seed
        self.key = str(uuid.uuid4())

        # 调用 /start 启动远端浏览器，并获取初始 observation
        start_resp = client_start(self.server_url, [self.question], [self.gt], [self.key])
        print("start_resp", start_resp)
        # 根据你的服务端逻辑，从返回的JSON中拿到初始内容
        # 比如你在服务端可能返回: {"obs": "...", "info": "...", ...}
        # 这里假定 server 返回 `{"observation": "...", "info": ...}` 供你使用:
        initial_obs = start_resp.get("obs", ["No observation from /start."])[0]
        # 也可能是 start_resp.get("render", "...")，要看你的后端如何返回

        # 重建对话 history
        self.history = [
            {"role": "system"},
            {
                "role": "user",
                "observation": initial_obs
            }
        ]

        print("=" * 30)
        print("[WikiQAEnv] reset done")
        print(f"Question: {self.question}")
        print(f"Ground Truth: {self.gt}")
        print(f"Start Observation: {initial_obs[:200]} ...")  # 截断显示
        print(f"Key: {self.key}, Index: {self.index}")
        print("=" * 30)

        return self.render()

    def reset_without_seed(self, question: str, gt: str, url: str) -> str:
        """
        reset 环境，不依赖本地 dataset seed，而是直接使用外部传入的 question, gt, url。
        同时将浏览器相关调用放到服务器端，与 reset(...) 保持一致。
        """

        # 1) 重置内部状态
        self.current_step = 0
        self.done = False
        self.reward = 0.0
        self.pred = None
        self.answer_similarity = 0.0
        self.answer_made = False

        # 2) 记录传入的 question / gt
        self.question = question
        self.gt = gt

        # 3) 生成新的 key（如果需要 index，可自行增加 self.index = ...）
        self.key = str(uuid.uuid4())

        # 4) 通过 client_start 调用远端服务，启动浏览器并获取初始观测
        #    注意这里需要扩展 client_start，让它能接受 url 或者你有其他方式传给后端
        start_resp = client_start(self.server_url, [self.question], [self.gt], [self.key])
        # 如果你需要传 url 过去，就把它加到 client_start 函数里，例如:
        # start_resp = client_start(self.server_url, [self.question], [self.gt], [self.key], [url])

        # 5) 从服务器返回的数据中取出初始 observation
        initial_obs = start_resp.get("obs", ["No observation from /start."])[0]

        # 6) 重建对话 history
        self.history = [
            {"role": "system"},
            {
                "role": "user",
                "question": self.question,
                "url": url,
                "observation": initial_obs
            }
        ]

        print("=" * 30)
        print("[WikiQAEnv] reset_without_seed done")
        print(f"Question: {self.question}")
        print(f"Ground Truth: {self.gt}")
        print(f"Start Observation: {initial_obs[:200]} ...")  # 截断显示
        print(f"Key: {self.key}")
        print("=" * 30)

        # 7) 返回一次 render()，与 reset(...) 的写法保持一致
        return self.render()

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        执行一步action:
          1. 解析动作 (extract_action)
          2. 判断是否为 STOP/CHECK 等终止动作, 若是则评估回答、给奖励、done=True
          3. 否则，发送请求到 /step，返回新的obs，拼接到 history
          4. 计算 reward，叠加到 self.reward

        Returns:
            observation (str): 当前环境的渲染 (给上层看看)
            reward (float): 该 step 获得的奖励
            done (bool): 是否结束
            info (dict): 额外信息 (可留空)
        """
        print("=" * 30)
        print(f"[WikiQAEnv] Step {self.current_step}")
        print(f"Question: {self.question}")
        print(f"GT: {self.gt}")
        print(f"Action: {action}")
        print("=" * 30)

        if self.done:
            return (self.render(), 0.0, True, {"action_is_effective": False})

        # 1. 先解析动作
        action_extracted, action_str = self.extract_action(action)
        # 2. 如果是 STOP/CHECK 动作，说明要结束了
        if action_extracted["action_type"] in [ActionTypes.STOP, ActionTypes.CHECK]:
            self.answer_made = True
            self.done = True
            self.pred = action_extracted['answer']  # 大模型提交的回答
            self.answer_similarity = fuzzy_match(self.gt, self.pred)
            format_reward = format_score(action, (action_extracted["action_type"] == ActionTypes.NONE))
            answer_reward = self.answer_similarity
            reward = -0.01 + format_reward + answer_reward
            self.reward += reward
            self.history.append({
                "role": "assistant",
                "pred": action,
                "reward": reward,
                "action_extracted": action_extracted
            })
            print(f"[STOP/CHECK] Answer: {self.pred}")
            print(f"[STOP/CHECK] Similarity: {self.answer_similarity:.3f}")
            print(f"[STOP/CHECK] Step Reward: {reward:.3f}, Total Reward: {self.reward:.3f}")
            return (self.render(), reward, self.done, {"action_is_effective": True})
        else:
            # 非 STOP 动作，先给个动作格式奖励
            format_reward = format_score(action, (action_extracted["action_type"] == ActionTypes.NONE))
            step_reward = -0.01 + format_reward
            self.reward += step_reward

            # 把该 step 先记录到 history
            self.history.append({
                "role": "assistant",
                "pred": action,
                "reward": step_reward,
                "action_extracted": action_extracted
            })

            # 3. 调用 /step，与服务端交互
            # 这里一次只发送1个 action，所以 query=[action], keys=[self.key]
            step_resp = client_step(self.server_url, [action], [self.key])
            # 服务端可返回 {"observation": "...", "if_done": bool, "info": {...}} 之类的
            new_obs = step_resp.get("obs", ["No observation from /step."])[0]
            server_done = step_resp.get("if_done", [False])[0]

            self.current_step += 1
            self.done = (self.current_step >= self.max_steps) or server_done

            # 记录新的 observation
            self.history.append({
                "role": "user",
                "observation": new_obs,
            })

            # 最后再检查一下本环境是否有什么自定义终止条件
            if self.check_break_condition():
                self.done = True

            print(f"[STEP] New obs snippet: {new_obs[:200]} ...")
            print(f"[STEP] Step Reward: {step_reward:.3f}, Total Reward: {self.reward:.3f}")
            return (self.render(), step_reward, self.done, {})

    def success(self) -> bool:
        """
        当回答相似度 >= threshold 则视为成功
        """
        return (self.answer_made and (self.answer_similarity >= self.threshold))

    def finished(self) -> bool:
        return self.done

    def render(self, prompt_format=None) -> str:
        """
        将 history 中的信息拼接到 prompt 中，返回给上层（通常是给大模型继续生成下一个动作）。
        prompt_format = "full" 或 "single" 等等
        """
        if prompt_format is None:
            prompt_format = self.prompt_format

        if prompt_format == "full":
            ans = ""
            for item in self.history:
                if item["role"] == "system":
                    ans += self.template_dict['system']
                elif item["role"] == "user":
                    ans += item["observation"]
                elif item["role"] == "assistant":
                    ans += self.template_dict['assistant'].format(pred=item["pred"])
                else:
                    raise ValueError("Unknown role in history")
            return ans + "<|im_start|>assistant"

        elif prompt_format == "single":
            # 只拿最后一轮用户信息 + system
            history = [self.history[0], self.history[-1]]
            ans = ""
            for item in history:
                if item["role"] == "system":
                    ans += self.template_dict['system']
                elif item["role"] == "user":
                    ans += item["observation"]
                elif item["role"] == "assistant":
                    ans += self.template_dict['assistant'].format(pred=item["pred"])
                else:
                    raise ValueError("Unknown role in history")
            return ans + "<|im_start|>assistant"

        else:
            raise NotImplementedError(f"Unsupported prompt_format={prompt_format}")

    def copy(self) -> "WikiQAEnv":
        """
        拷贝自身，注意此时如果需要在新副本中也跑远端环境，
        最好重新分配新的 key / index，否则可能相互冲突。
        这里演示的是浅/深拷贝结合的示例，你可按需自定义。
        """
        new_instance = self.__class__.__new__(self.__class__)

        for attr, value in self.__dict__.items():
            if attr in ["dataset", "server_url"]:
                setattr(new_instance, attr, value)  # 直接引用
            else:
                setattr(new_instance, attr, deepcopy(value))  # 深拷贝

        # 重新分配新的 key/index，以免冲突
        new_instance.key = str(uuid.uuid4())
        new_instance.index = 10000 + random.randint(0, 9999)

        return new_instance

    # ============== Tool Functions ==============
    def extract_action(self, response: str):
        """
        解析大模型输出中的动作标记，比如:
          <think>xxx</think>\n```stop [some answer]```
        """
        force_prefix = self.prompt_constructor.instruction["meta_data"].get("force_prefix", "")
        response = f"{force_prefix}{response}"

        try:
            parsed_response = self.prompt_constructor.extract_action(response)
            action = create_id_based_action(parsed_response)
            action["raw_prediction"] = response
        except ActionParsingError as e:
            print(f"ActionParsingError: {e}")
            # 无法解析就当 none action
            action = create_none_action()
            parsed_response = "The action is invalid, please retry"
        return action, parsed_response

    def check_break_condition(self):
        """
        如果你有额外的退出条件，可在这里判断。
        现在默认不做任何额外判断，返回 False。
        """
        return False

# ============ 测试示例 ============
def test_wiki_qa_env():
    import pandas as pd
    from copy import deepcopy

    # 1. 读取数据集
    dataset_path = "/home/zhiheng/WikiRL/ragen/env/wiki/data/puzzle/test.parquet"
    df = pd.read_parquet(dataset_path)

    # 2. 实例化环境，指定 seed=1
    env = WikiQAEnv(dataset=df, seed=1)
    # 3. reset 环境，打印初始渲染结果
    print("=== Initial Reset & Render ===")
    observation = env.reset()
    print(observation)

    # 4. 第一次 step: 执行 'goto [google.com]' 并打印渲染结果
    action_1 = "<think>balabalabalabala</think>\n```click [99]```"
    print("=== Step 1: Action ===")
    print(f"Action: {action_1}")
    observation, reward, done, info = env.step(action_1)
    print("=== Observation After Action 1 ===")
    print(env.render())
    print(f"Reward: {reward}, Done: {done}")
    print("--------------------------------------------------")

    # 5. 第二次 step: 执行 'stop [I don't know.]' 并打印渲染结果
    action_2 = "<think>balabalabalabala</think>\n```stop [Sept 18, 2018]```"
    print("=== Step 2: Action ===")
    print(f"Action: {action_2}")
    observation, reward, done, info = env.step(action_2)
    print("=== Observation After Action 2 ===")
    print(env.render())
    print(f"Reward: {reward}, Done: {done}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    test_wiki_qa_env()
