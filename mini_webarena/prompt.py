import json
import re
from pathlib import Path
from typing import Any, TypedDict

from .browser_env import Trajectory
from .browser_actions import Action, ActionParsingError
from .browser_login import URL_MAPPINGS
from .utils import StateInfo, LMConfig
from .model import Tokenizer
# from llms.utils import APIInput


class Instruction(TypedDict):
    """Instruction for constructing prompt"""

    intro: str
    examples: list[tuple[str, str]]
    template: str
    meta_data: dict[str, Any]

instruction_path = Path(__file__).parent / "prompt.json"

class PromptConstructor(object):
    def __init__(
            self,
            # instruction_path: str | Path,
            lm_config: LMConfig,
            tokenizer: Tokenizer,
    ):
        # self.instruction_path = Path(instruction_path)
        self.obs_modality = "text"
        self.lm_config = lm_config
        # load from json_prompt
        instruction = json.load(open(instruction_path))
        instruction["examples"] = [tuple(e) for e in instruction["examples"]]
        self.instruction: Instruction = instruction
        self.tokenizer = tokenizer

    def get_lm_api_input(
            self, intro: str, examples: list[tuple[str, str]], current: str
    ):
        """Return the require format for an API"""
        message: list[dict[str, str]] | str
        # if "openai" in self.lm_config.provider:
        #     if self.lm_config.mode == "chat":
        #         message = [{"role": "system", "content": intro}]
        #         for (x, y) in examples:
        #             message.append(
        #                 {
        #                     "role": "system",
        #                     "name": "example_user",
        #                     "content": x,
        #                 }
        #             )
        #             message.append(
        #                 {
        #                     "role": "system",
        #                     "name": "example_assistant",
        #                     "content": y,
        #                 }
        #             )
        #         message.append({"role": "user", "content": current})
        #         return message
        #     elif self.lm_config.mode == "completion":
        #         message = f"{intro}\\n\\n"
        #         message += "Here are a few examples:\\n"
        #         for example in examples:
        #             message += f"Observation\\n:{example[0]}\\n\\n"
        #             message += f"Action: {example[1]}\\n\\n"
        #         message += "Now make prediction given the observation\\n\\n"
        #         message += f"Observation\\n:{current}\\n\\n"
        #         message += "Action:"
        #         return message
        #     else:
        #         raise ValueError(
        #             f"OpenAI models do not support mode {self.lm_config.mode}"
        #         )
        # elif "huggingface" in self.lm_config.provider:
        if "huggingface" in self.lm_config.provider:
            # https://huggingface.co/blog/llama2#how-to-prompt-llama-2
            # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L320
            if "Llama-2" in self.lm_config.model:
                if self.lm_config.mode == "chat":
                    B_INST, E_INST = "[INST]", "[/INST]"
                    B_SYS, E_SYS = "<<SYS>>\\n", "\\n<</SYS>>\\n\\n"
                    BOS, EOS = "<s>", "</s>"
                    # adding the system message to be the starting of the first example
                    examples = [
                                   (
                                       B_SYS + intro + E_SYS + examples[0][0],
                                       examples[0][1],
                                   )
                               ] + examples[1:]
                    message = "".join(
                        [
                            f"{BOS}{B_INST} {x.strip()} {E_INST} {y.strip()} {EOS}"
                            for (x, y) in examples
                        ]
                    )
                    # add the current observation
                    message += f"{BOS}{B_INST} {current.strip()} {E_INST} {self.instruction['meta_data'].get('force_prefix', '')}"

                    return message
                else:
                    raise ValueError("Only chat mode is supported for Llama-2")
            elif "Qwen" in self.lm_config.model:
                if self.lm_config.mode == "chat":
                        B_INST, E_INST = "<|im_start|>user\\n", "<|im_end|>\\n"
                        B_SYS, E_SYS = "<|im_start|>system\\n", "<|im_end|>\\n"
                        BOS, EOS = "<|im_start|>assistant\\n", "<|im_end|>\\n"

                        # 添加system message到第一个example
                        examples = [
                            (
                                B_SYS + intro + E_SYS + examples[0][0],
                                examples[0][1],
                            )
                        ] + examples[1:]

                        message = "".join(
                            [
                                f"{B_INST}{x.strip()}{E_INST}{BOS}{y.strip()}{EOS}"
                                for (x, y) in examples
                            ]
                        )

                        # 添加当前输入
                        message += f"{B_INST}{current.strip()}{E_INST}{BOS}{self.instruction['meta_data'].get('force_prefix', '')}"

                        return message
            else:
                raise ValueError(
                    f"Huggingface models do not support model_tag {self.lm_config.gen_config['model_tag']}"
                )
        # elif "ours" in self.lm_config.provider:
        #     message = f"{intro}\\n\\n"
        #     message += "Now make prediction given the observation\\n\\n"
        #     message += f"Observation\\n:{current}\\n\\n"
        #     message += "Action:"
        #     return message
        else:
            raise NotImplementedError(
                f"Provider {self.lm_config.provider} not implemented"
            )

    def construct(
            self,
            trajectory: Trajectory,
            intent: str,
            meta_data: dict[str, Any] = {},
    ):
        raise NotImplementedError

    def map_url_to_real(self, url: str) -> str:
        """Map the urls to their real world counterparts"""
        for i, j in URL_MAPPINGS.items():
            if i in url:
                url = url.replace(i, j)
        return url

    def map_url_to_local(self, url: str) -> str:
        """Map the urls to their local counterparts"""
        for i, j in URL_MAPPINGS.items():
            if j in url:
                url = url.replace(j, i)
            # https
            if j.replace("http", "https") in url:
                url = url.replace(j.replace("http", "https"), i)
        return url

    def _extract_action(self, response: str) -> str:
        raise NotImplementedError

    def extract_action(self, response: str) -> str:
        response = self._extract_action(response)
        response = self.map_url_to_local(response)
        return response

class CoTPromptConstructor(PromptConstructor):
    """The agent will perform step-by-step reasoning before the answer"""

    def __init__(
            self,
            # instruction_path: str | Path,
            lm_config: LMConfig,
            tokenizer: Tokenizer,
    ):
        super().__init__(lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]
        self.max_length = 4096
        self.tokenizer = tokenizer

    def construct(
            self,
            trajectory: Trajectory,
            intent: str,
            meta_data: dict[str, Any] = {},
    ):
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        # print(self.lm_config)
        # exit(0)
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        page = state_info["info"]["page"]
        url = page.url
        previous_action_str = meta_data["action_history"][-1]
        current = template.format(
            objective=intent,
            url=url,
            observation=obs,
            previous_action=previous_action_str,
        )

        assert all([f"{{k}}" not in current for k in keywords])

        prompt = self.get_lm_api_input(intro, examples, current)
        return prompt

    # def _extract_action(self, response: str) -> str:
    #     # find the first occurence of action
    #     action_splitter = self.instruction["meta_data"]["action_splitter"]
    #     pattern = rf"{action_splitter}((.|\\n)*?){action_splitter}"
    #     match = re.search(pattern, response)
    #     if match:
    #         return match.group(1).strip()
    #     else:
    #         preview = response[:300].replace("\n", "\\n") + ("..." if len(response) > 300 else "")
    #         raise ActionParsingError(
    #             f"Failed to parse action using splitter `{action_splitter}`. "
    #             f"No match found in the response.\n"
    #             f"Response preview: \"{preview}\"\n"
    #             f"Expected format: `{action_splitter}<action>{action_splitter}`"
    #         )
    def _extract_action(self, response: str) -> str:
        # 先尝试新格式
        match = re.search(r"<action>(.*?)</action>", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # 再尝试旧格式
        match = re.search(r"```(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        preview = response[:300].replace("\n", "\\n") + ("..." if len(response) > 300 else "")
        raise ActionParsingError(
            "Failed to parse action using <action>...</action> or ```...```.\n"
            f"Response preview: \"{preview}\"\n"
            "Expected format: <action>your_action_here</action> or ```your_action_here```"
        )

if __name__ == "__main__":
    pass