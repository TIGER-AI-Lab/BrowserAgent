import argparse
import json
from typing import Any

import tiktoken
from beartype import beartype

from .prompt import *
from .browser_env import Trajectory
from .browser_actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
)
from .utils import Observation, StateInfo, LMConfig, construct_llm_config

# Refactored Implementation
from .model_sglang import Tokenizer, call_llm

class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        pass

    def next_action(
            self, trajectory: Trajectory, intent: str, meta_data: Any
    ):
        """Predict the next action given the observation"""
        raise NotImplementedError

    def check_action(
            self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any], target_action: str
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
            self,
            test_config_file: str,
    ) -> None:
        raise NotImplementedError

class PromptAgent(Agent):
    """prompt-based agent that emits action given the history"""

    @beartype
    def __init__(
            self,
            action_set_tag: str,
            lm_config: LMConfig,
            prompt_constructor: PromptConstructor,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    @beartype
    def next_action(
            self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any]
    ):
        # TODO: also return the action that not successfully parsed
        prompt = self.prompt_constructor.construct(
            trajectory, intent, meta_data
        ) # + "Let's think step-by-step. This page list the information of "

        lm_config = self.lm_config
        n = 0
        failed_actions = []
        while True:
            response = call_llm(lm_config, prompt)
            force_prefix = self.prompt_constructor.instruction[
                "meta_data"
            ].get("force_prefix", "")
            response = f"{force_prefix}{response}"
            n += 1
            # print(self.action_set_tag)
            action = parse_action(response, self.prompt_constructor)
            if action["action_type"] != 0 or n >= lm_config.gen_config["max_retry"]:
                break
            else:
                failed_actions.append(action)
            # try:
            #     parsed_response = self.prompt_constructor.extract_action(
            #         response
            #     )
            #     if self.action_set_tag in ["id_html_tree", "id_html_nasc_tree", "id_accessibility_tree"]:
            #         action = create_id_based_action(parsed_response)
            #     elif self.action_set_tag == "playwright":
            #         action = create_playwright_action(parsed_response)
            #     else:
            #         raise ValueError(
            #             f"Unknown action type {self.action_set_tag}"
            #         )
            #     action["raw_prediction"] = response
            #     break
            # except ActionParsingError as e:
            #     action = create_none_action()
            #     action["raw_prediction"] = response
            #     if n >= lm_config.gen_config["max_retry"]:
            #         break
            #     else:
            #         failed_actions.append(action)

        return action, failed_actions

    def check_action( # Assume this function is useless
            self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any], target_action: str
    ) -> Action:
        return None
        # prompt = self.prompt_constructor.construct(
        #     trajectory, intent, meta_data
        # )
        # lm_config = self.lm_config
        # n = 0
        #
        # # agent will retry if the action is not parsed correctly
        # while True:
        #     response = target_action
        #     force_prefix = self.prompt_constructor.instruction[
        #         "meta_data"
        #     ].get("force_prefix", "")
        #     response = f"{force_prefix}{response}"
        #     n += 1
        #     try:
        #         parsed_response = self.prompt_constructor.extract_action(
        #             response
        #         )
        #         if self.action_set_tag in ["id_accessibility_tree", "id_html_tree", "id_html_nasc_tree"]:
        #             action = create_id_based_action(parsed_response)
        #         elif self.action_set_tag == "playwright":
        #             action = create_playwright_action(parsed_response)
        #         else:
        #             raise ValueError(
        #                 f"Unknown action type {self.action_set_tag}"
        #             )
        #         action["raw_prediction"] = response
        #         break
        #     except ActionParsingError as e:
        #         if n >= lm_config.gen_config["max_retry"]:
        #             action = create_none_action()
        #             action["raw_prediction"] = response
        #             break
        #
        # return prompt, action

    def reset(self, test_config_file: str) -> None:
        pass

def parse_action(response, prompt_constructor):
    try:
        parsed_response = prompt_constructor.extract_action(response)
        action = create_id_based_action(parsed_response)
        action["raw_prediction"] = response
    except ActionParsingError as e:
        action = create_none_action()
        action["raw_prediction"] = response
    # print(action)
    return action

# def construct_agent(model_name, inference_endpoint) -> Agent:
#     llm_config = construct_llm_config(model_name, inference_endpoint)
#     tokenizer = Tokenizer("huggingface", model_name)
#     from prompt import CoTPromptConstructor
#     prompt_constructor = CoTPromptConstructor(
#             "/home/zhiheng/AgentRAG/agent/jsons/p_cot_id_actree_2s.json", lm_config=llm_config, tokenizer=tokenizer
#         )
#     agent = PromptAgent(
#             action_set_tag="id_accessibility_tree",
#             lm_config=llm_config,
#             prompt_constructor=prompt_constructor,
#         )
#     return agent

# Once we have this we can decompose the agent into smaller components
def construct_promptConstructor(model_name, inference_endpoint) -> PromptConstructor:
    llm_config = construct_llm_config(model_name, inference_endpoint)
    tokenizer = Tokenizer("huggingface", model_name)
    prompt_constructor = CoTPromptConstructor(lm_config=llm_config, tokenizer=tokenizer)
    return prompt_constructor, tokenizer, llm_config