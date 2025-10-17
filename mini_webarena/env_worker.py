# import nest_asyncio
# nest_asyncio.apply()
# import asyncio
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


class WikiQAEnv(object):
    def __init__(
            self,
            question, gt,
            max_steps: int = 10,
            threshold: float = 0.7,
            prompt_format="single",  # full, last, single, tunc
            url = None
    ):
        super().__init__()
        self.max_steps = max_steps
        self.threshold = threshold
        self.prompt_format = prompt_format
        self.current_step = 0
        self.done = False
        self.question = question
        self.gt = gt
        self.pred = None
        self.obs_modality = "text"

        from .create_dataset import TEMPLATES
        self.template_dict = TEMPLATES['qwen-instruct']

        self.env = ScriptBrowserEnv(
            headless=True,
            slow_mo=0,
            observation_type="accessibility_tree",
            current_viewport_only=True,
            viewport_size={"width": 1280, "height": 720},
            save_trace_enabled=True,
            sleep_after_execution=0.0,
            simple_mode=True,
            page_load_timeout=60.0  # 增加到60秒等待时间
        )

        from .agent import construct_promptConstructor
        self.prompt_constructor, self.tokenizer, _ = construct_promptConstructor("Qwen/Qwen2.5-14B-Instruct", None)
        if url == None:
            self.url = "https://tigerai.ca/wiki/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
        else:
            self.url = url
        obs, _ = self.env.reset_without_config(start_url=self.url)
        self.history = [{"role": "system"}, {"role": "user", "question": self.question, "url": self.url,
                                             "observation": obs[self.obs_modality], "previous_action": None}]
        # self.pure_obs_temp = ("<browser>Objective: {objective}\n\n"
        # "URL: {url}\n"
        # "Observation:\n"
        # "{observation}\n"
        # "Parsed Previous Action:\n"
        # "{previous_action}\n</browser>")
        self.pure_obs_temp = ("Objective: {objective}\n\n"
        "URL: {url}\n"
        "Observation:\n"
        "{observation}\n"
        "Parsed Previous Action:\n"
        "{previous_action}\n")


    def __str__(self):
        return f"WikiQAEnv(seed={self.seed}, question={self.question}, answer={self.gt})"

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Returns:
            observation (str): 当前环境渲染
            done (bool): 是否结束
            # info (dict): 额外信息
            validity (bool): 动作是否有效
        """
        validity = True
        terminated = False
        if self.done:
            return (self.render(), True, False)
        action_extracted, action_str = self.extract_action(action)
        if action_extracted["action_type"] == ActionTypes.CHECK or action_extracted["action_type"] == ActionTypes.STOP:
            self.answer_made = True
            self.done = True
            # self.pred = action_extracted['answer']
            obs = self.render()
            self.history.append(
                {"role": "assistant", "pred": action, "action_extracted": action_extracted})
            return (obs, self.done, validity)

        if action_extracted["action_type"] == ActionTypes.NONE:
            validity = True
            action_extracted = create_none_action()
            # 确保页面状态稳定后再获取观察结果
            self.env._wait_for_page_ready()
            obs = self.env._get_obs()
        else:
            try:
                obs, _, terminated, _, _ = self.env.step(action_extracted)
                # step方法内部已经调用了_wait_for_page_ready，这里不需要重复调用
            except Exception as e:
                print("######################### Error in run step, action invalid")
                print(action_extracted)
                action_extracted = create_none_action()
                # 确保页面状态稳定后再获取观察结果
                self.env._wait_for_page_ready()
                obs = self.env._get_obs()
                validity = False

        self.history.append(
            {"role": "assistant", "pred": action, "action_extracted": action_extracted})

        self.current_step += 1
        self.done = self.current_step >= self.max_steps or terminated
        self.url = self.env.page.url
        self.history.append({"role": "user", "question": self.question, "url": self.url, "observation": obs[self.obs_modality],
             "previous_action": action_str})
        observation = self.render()
        return (observation, self.done, validity)

    def finished(self) -> bool:
        return self.done

    def render(self, prompt_format=None) -> str:
        if prompt_format is None:
            prompt_format = self.prompt_format

        if prompt_format == "full":
            ans = ""
            for item in self.history:
                if item["role"] == "system":
                    ans += self.template_dict['system']
                elif item["role"] == "user":
                    ans += self.template_dict['user'].format(objective=item["question"], url=item["url"], observation
                    =item["observation"], previous_action=item["previous_action"])
                elif item["role"] == "assistant":
                    ans += self.template_dict['assistant'].format(pred=item["pred"])
                else:
                    raise ValueError("role not recognized")
            return ans + "<|im_start|>assistant"
        elif prompt_format == "single":
            history = [self.history[0], self.history[-1]]
            ans = ""
            for item in history:
                if item["role"] == "system":
                    ans += self.template_dict['system']
                    # raise ValueError("role not recognized")
                elif item["role"] == "user":
                    ans += self.template_dict['user'].format(objective=item["question"], url=item["url"], observation
                    =item["observation"], previous_action=item["previous_action"])
                    # ans += self.pure_obs_temp.format(objective=item["question"], url=item["url"], observation=item["observation"],
                    #                                  previous_action=item["previous_action"])
                elif item["role"] == "assistant":
                    ans += self.template_dict['assistant'].format(pred=item["pred"])
                    # raise ValueError("role not recognized")
                else:
                    raise ValueError("role not recognized")
            return ans + "<|im_start|>assistant"
        elif prompt_format == "last":
            history = [self.history[-1]]
            ans = ""
            for item in history:
                if item["role"] == "system":
                    # ans += self.template_dict['system']
                    print(item)
                    raise ValueError("role not recognized")
                elif item["role"] == "user":
                    ans += self.pure_obs_temp.format(objective=item["question"], url=item["url"], observation=item["observation"],
                                                     previous_action=item["previous_action"])
                else:
                    print(item)
                    raise ValueError("role not recognized")
            return ans
        else:
            raise NotImplementedError

    def close(self):
        try:
            if hasattr(self, "env") and self.env is not None:
                self.env.close()
        except Exception as e:
            print("Error closing environment:", e)

    # ============== Tool Functions ==============
    def extract_action(self, response: str):
        if response == "" or response == None:
            return create_none_action(), ""
        force_prefix = self.prompt_constructor.instruction[
            "meta_data"
        ].get("force_prefix", "")
        response = f"{force_prefix}{response}"
        try:
            parsed_response = self.prompt_constructor.extract_action(
                response
            )
            action = create_id_based_action(parsed_response)
            action["raw_prediction"] = response
        except ActionParsingError as e:
            print(f"ActionParsingError: {e}")
            action = create_none_action()
            parsed_response = "The action is invalid, please retry"
        return action, parsed_response

def test_wiki_qa_env():
    import time
    # 1. 实例化环境
    env = WikiQAEnv("Who is current US president", "Biden", prompt_format="single")

    # action_1 = (
    #             '<think>To find out who plays the wildling woman in "Game of Thrones," '
    #             'I should use the search functionality on this page. The search textbox has the ID 21. '
    #             'I will type the query "Game of Thrones wildling woman actor" into the search box and press enter to search for the information.</think>               \n'
    #             '```type [16] [Game of Thrones wildling woman actor] [1]```�\']>;\n<thէ'
    #         ),
    action_1 ="""The search box is available for typing queries. I should type ""Deadpool"" into it to find information about the movie.  
```type [16] Deadpool [press_enter_after=1]```"""
    print("=== Step 1: Action ===") 
    print("Action:", None)
    observation, done, validity = env.step(None)
    print("=== Observation After Action 1 ===")
    print("Render:", env.render())
    print(f"Done: {done}, Validity: {validity}")
    print("--------------------------------------------------")

    print("=== Step 2: Repeat Action ===")
    print("Action:", action_1)
    observation, done, validity = env.step(action_1)
    print("=== Observation After Repeat Action 1 ===")
    # print("Render:", env.render())
    print(f"Done: {done}, Validity: {validity}")
    print("--------------------------------------------------")

    # 4. 第二次 step: 执行 action_2
    action_2 = "<think>balabalabalabala</think>          \n<action>stop [down]</action>"
    print("=== Step 2: Action ===")
    print("Action:", action_2)
    observation, done, validity = env.step(action_2)
    print("=== Observation After Action 2 ===")
    # print("Render:", env.env._get_obs())
    print(f"Done: {done}, Validity: {validity}")
    print("--------------------------------------------------")

    # 最后关闭环境
    env.close()

if __name__ == "__main__":
    test_wiki_qa_env()
