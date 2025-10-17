# import nest_asyncio
# nest_asyncio.apply()
import asyncio
import re
import random
from copy import deepcopy
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

# 假设你的 BaseLanguageBasedEnv 就在同目录下的 base.py
from .env_base import BaseLanguageBasedEnv
from .browser_actions import (
    Action,
    ActionTypes,
    create_stop_action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
)
from .browser_env import ScriptBrowserEnv, Trajectory
# from .browser_env_async import AsyncScriptBrowserEnv

class WikiQAEnv(BaseLanguageBasedEnv):
    def __init__(
        self,
        dataset = None,  # a huggingface dataset
        seed: int = 0, # Using which row of dataset
        max_steps: int = 10,
        threshold: float = 0.7,
        prompt_format = "full", # full, last, single
        browser_api = "sync", # sync, async
    ):
        """
        Args:
            dataset: 包含多条问答数据的 DataFrame, 假设有列 'question' 和 'answer'
            seed: 用来选取第几条问答
            max_steps: 最大交互步数，超限后强制结束
            threshold: 当 final answer 与真实答案的相似度 >= threshold 则视为成功
        """

        print("=" * 30)
        print("WikiQAEnv init")

        super().__init__()
        self.dataset = dataset
        self.seed = seed
        self.max_steps = max_steps
        self.threshold = threshold
        self.prompt_format = prompt_format
        self.browser_api = browser_api
        # print("[DEBUG] WikiQAEnv init Checkpoint 1")

        self.current_step = 0
        self.done = False
        self.reward = 0.0
        # print("[DEBUG] WikiQAEnv init Checkpoint 2")
        # get the dict of the dataset {seed}-th row from pd.df
        datapoint = self.dataset.iloc[self.seed]  # 按行号索引
        # print(datapoint)
        # print(datapoint.keys())
        # print(datapoint['extra_info'].keys())
        # exit(1)
        self.question = datapoint["extra_info"]['question']
        # print("[DEBUG] WikiQAEnv init Checkpoint 3")
        self.gt = datapoint["extra_info"]['golden_answers']

        print(self.question)
        print(self.gt)

        self.pred = None
        # TODO, heed to change to multiple answers
        self.answer_similarity = 0.0
        self.answer_made = False  # 是否已做出回答
        self.obs_modality = "text"

        from .create_dataset import TEMPLATES
        self.template_dict = TEMPLATES['qwen-instruct']
        # print("[DEBUG] WikiQAEnv init Checkpoint 4")

        # Web Browser Environment
        if self.browser_api == "async":
            # self.env = AsyncScriptBrowserEnv(
            #     headless=True,
            #     slow_mo=0,
            #     observation_type="accessibility_tree",
            #     current_viewport_only=True,
            #     viewport_size={"width": 1280, "height": 720},
            #     save_trace_enabled=True,
            #     sleep_after_execution=0.0,
            # )
            raise NotImplementedError
        else:
            self.env = ScriptBrowserEnv(
                headless=True,
                slow_mo=0,
                observation_type="accessibility_tree",
                current_viewport_only=True,
                viewport_size={"width": 1280, "height": 720},
                save_trace_enabled=True,
                sleep_after_execution=0.0,
                page_load_timeout=60.0  # 增加到60秒等待时间
            )

        # print("[DEBUG] WikiQAEnv init Checkpoint 5")
        from .agent import construct_promptConstructor
        self.prompt_constructor, self.tokenizer, _ = construct_promptConstructor("Qwen/Qwen2.5-14B-Instruct", None)
        self.url = "https://tigerai.ca/wiki/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"

        # print("[DEBUG] WikiQAEnv init Checkpoint 6")
        if self.browser_api == "async":
            obs, info = asyncio.run(self.env.reset_without_config(start_url=self.url))
        else:
            obs, info = self.env.reset_without_config(start_url=self.url)
        # reset_without_config方法内部已经调用了_wait_for_page_ready，确保页面加载完成

        # print("[DEBUG] WikiQAEnv init Checkpoint 7")
        self.history = [{"role": "system"}, {"role": "user", "question": self.question, "url": self.url,
                                             "observation": obs[self.obs_modality], "previous_action": None}]

        # print("[DEBUG] WikiQAEnv init Checkpoint 8")
        # 重置跟踪变量
        self._reset_tracking_variables()

        print("=" * 30)

    def __str__(self):
        return f"WikiQAEnv(seed={self.seed}, question={self.question}, answer={self.gt})"

    def reset(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            self.seed = seed

        # Reset internal state
        self.current_step = 0
        self.done = False

        # Re-pick data from the dataset
        datapoint = self.dataset.iloc[self.seed]
        self.question = datapoint["extra_info"]["question"]
        self.gt = datapoint["extra_info"]["gt"]
        self.pred = None
        self.answer_similarity = 0.0
        self.answer_made = False

        # Reset the browser environment
        if self.browser_api == "async":
            obs, info = asyncio.run(self.env.reset_without_config(start_url=self.url))
        else:
            obs, info = self.env.reset_without_config(start_url=self.url)
        # reset_without_config方法内部已经调用了_wait_for_page_ready，确保页面加载完成

        # Rebuild initial history
        self.history = [
            {"role": "system"},
            {
                "role": "user",
                "question": self.question,
                "url": self.url,
                "observation": obs[self.obs_modality],
                "previous_action": None
            }
        ]

        # Reset any custom tracking variables
        self._reset_tracking_variables()

        print("=" * 30)
        print("WikiQAEnv reset")
        print(self.question)
        print(self.gt)
        print("=" * 30)

        # Return whatever the render() method produces
        return self.render()

    def reset_without_seed(self, question: str, gt: str, url: str) -> str:
        self.current_step = 0
        self.done = False

        self.question = question
        self.gt = gt
        self.pred = None
        self.answer_similarity = 0.0
        self.answer_made = False

        if self.browser_api == "async":
            obs, info = asyncio.run(self.env.reset_without_config(start_url=url))
        else:
            obs, info = self.env.reset_without_config(start_url=url)
        # reset_without_config方法内部已经调用了_wait_for_page_ready，确保页面加载完成

        self.history = [
            {"role": "system"},
            {
                "role": "user",
                "question": self.question,
                "url": url,
                "observation": obs[self.obs_modality],
                "previous_action": None
            }
        ]

        self._reset_tracking_variables()

        print("=" * 30)
        print("WikiQAEnv reset")
        print(self.question)
        print(self.gt)
        print("=" * 30)

        return self.render()

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Returns:
            observation (str): 当前环境渲染
            reward (float): 本step得到的奖励
            done (bool): 是否结束
            info (dict): 额外信息
        """
        print("=" * 30)
        print("WikiQAEnv Step")
        print(self.question)
        print(self.gt)
        print(action)
        print("=" * 30)
        if self.done:
            return (self.render(), 0.0, True, {"action_is_effective": False})
        # TODO
        # Step 1. check if done, calculate reward, append history
        action_extracted, action_str = self.extract_action(action)
        from .rl_utils import format_score
        from .evaluator import fuzzy_match
        is_success = action_extracted["action_type"] == ActionTypes.NONE
        format_reward = format_score(action, is_success)
        if action_extracted["action_type"] == ActionTypes.CHECK or action_extracted["action_type"] == ActionTypes.STOP:
            self.answer_made = True
            self.done = True
            # self.pred = action_extracted["raw_prediction"]
            self.pred = action_extracted['answer']
            # print(action_extracted)
            # print("#"*100)
            self.answer_similarity = fuzzy_match(self.gt, action_extracted['answer'])
            answer_reward = self.answer_similarity
            reward = -0.01 + format_reward + answer_reward
            self.reward += reward
            self.history.append(
                {"role": "assistant", "pred": action, "reward": reward, "action_extracted": action_extracted})
            print(f"Format Reward: {format_reward}, Answer Reward: {answer_reward}, Total Reward: {reward}")
            return (self.render(), reward, self.done, {"action_is_effective": True})

        reward = -0.01 + format_reward
        self.reward += reward
        self.history.append({"role": "assistant", "pred": action, "reward": reward, "action_extracted": action_extracted})
        # Step 2. execute action
        print(action_extracted)
        obs, _, terminated, _, info = self.env.step(action_extracted)
        # step方法内部已经调用了_wait_for_page_ready，确保页面加载完成
        self.current_step += 1
        self.done = self.current_step >= self.max_steps or terminated
        # Step 3. add observation to history
        print(obs)
        print(terminated)
        self.history.append({"role": "user", "question": self.question, "url": self.url, "observation": obs[self.obs_modality],
                             "previous_action": action_str})
        observation = self.render()
        # print()
        # print(observation)
        # exit(0)
        # Step 4. check if meet the break condition
        if self.check_break_condition():
            self.done = True
        return (observation, reward, self.done, info)

    def success(self) -> bool:
        """
        当做出回答 且 相似度>=threshold 就算成功
        # Maybe we can also use exact match, or other method from evaluator@fuzzy_match; TODO
        """
        return (self.answer_made and (self.answer_similarity >= self.threshold))
    #
    def finished(self) -> bool:
        return self.done

    def render(self, prompt_format = None) -> str:
        if prompt_format == None:
            prompt_format = self.prompt_format
        if prompt_format == "full":
            ans = ""
            print("#"*100)
            print(len(self.history))
            for item in self.history:
                if item["role"] == "system":
                    ans += self.template_dict['system']
                elif item["role"] == "user":
                    ans += self.template_dict['user'].format(objective = item["question"], url = item["url"], observation
                    = item["observation"], previous_action = item["previous_action"])
                elif item["role"] == "assistant":
                    ans += self.template_dict['assistant'].format(pred = item["pred"])
                else:
                    raise ValueError("role not recognized")
            return ans + "<|im_start|>assistant"
        elif prompt_format == "single":
            history = [self.history[0], self.history[-1]]
            ans = ""
            for item in history:
                if item["role"] == "system":
                    ans += self.template_dict['system']
                elif item["role"] == "user":
                    ans += self.template_dict['user'].format(objective = item["question"], url = item["url"], observation
                    = item["observation"], previous_action = item["previous_action"])
                elif item["role"] == "assistant":
                    ans += self.template_dict['assistant'].format(pred = item["pred"])
                else:
                    raise ValueError("role not recognized")
            return ans + "<|im_start|>assistant"
        else:
            raise NotImplementedError

    def copy(self) -> "WikiQAEnv":
        # 创建一个新实例
        new_instance = self.__class__.__new__(self.__class__)

        print("=" * 30)
        print("WikiQAEnv copy")
        print(self.question)
        print(self.gt)
        print("=" * 30)
        # 复制所有属性
        for attr, value in self.__dict__.items():
            if attr == "dataset":
                setattr(new_instance, attr, value)  # 直接引用 dataset（浅拷贝）
            elif attr == "env":
                if self.browser_api == "async":
                    # new_value = AsyncScriptBrowserEnv(
                    #     headless=True,
                    #     slow_mo=0,
                    #     observation_type="accessibility_tree",
                    #     current_viewport_only=True,
                    #     viewport_size={"width": 1280, "height": 720},
                    #     save_trace_enabled=True,
                    #     sleep_after_execution=0.0,
                    # )
                    # asyncio.run(new_value.reset_without_config(start_url=self.url))
                    raise NotImplementedError
                else:
                    raise ValueError
                setattr(new_instance, attr, new_value)
            else:
                setattr(new_instance, attr, deepcopy(value))  # 其他变量深拷贝
        return new_instance

    # ============== Tool Functions ==============
    def extract_action(self, response: str):
        force_prefix = self.prompt_constructor.instruction[
            "meta_data"
        ].get("force_prefix", "")
        response = f"{force_prefix}{response}"
        try:
            parsed_response = self.prompt_constructor.extract_action(
                response
            )
            print(parsed_response)
            action = create_id_based_action(parsed_response)
            print(action)
            # exit(0)
            action["raw_prediction"] = response
        except ActionParsingError as e:
            print(f"ActionParsingError: {e}")
            # raise e
            action = create_none_action()
            parsed_response = "The action is invalid, please retry"
        return action, parsed_response

    def check_break_condition(self):
        # If something happen in the history, then return -1
        # TODO
        return False

# ============ 测试示例 ============
def test_wiki_qa_env():
    import pandas as pd
    from copy import deepcopy

    # 1. 读取数据集
    dataset_path = "/data/minimax-dialogue/users/ruobai/rl_r2e/data/wikiQA_debug/dev.parquet"
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
    action_2 = "<think>balabalabalabala</think>\n```type [1407] [death row inmates in the US] [1]```"
    print("=== Step 2: Action ===")
    print(f"Action: {action_2}")
    observation, reward, done, info = env.step(action_2)
    print("=== Observation After Action 2 ===")
    print(env.render())
    print(f"Reward: {reward}, Done: {done}")
    print("--------------------------------------------------")

    # 5. 第二次 step: 执行 'stop [I don't know.]' 并打印渲染结果
    action_3 = "<think>balabalabalabala</think>\n```stop [Sept 18, 2018]```"
    print("=== Step 3: Action ===")
    print(f"Action: {action_3}")
    observation, reward, done, info = env.step(action_3)
    print("=== Observation After Action 3 ===")
    print(env.render())
    print(f"Reward: {reward}, Done: {done}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    test_wiki_qa_env()
