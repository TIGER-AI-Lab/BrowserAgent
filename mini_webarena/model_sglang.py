# tokenizers.py
from typing import Any
from transformers import AutoTokenizer  # type: ignore

class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        # if provider == "openai":
        #     self.tokenizer = tiktoken.encoding_for_model(model_name)
        # elif provider == "huggingface":
        if provider == "huggingface":
            # print(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # turn off adding special tokens automatically
            self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
            self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
            self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]
        # elif provider == "ours":
        #     self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        else:
            raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

# Adapt from sglang in soup
import requests
import torch
import os

from typing import Any
# from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import time

# # 加载 SentenceTransformer 模型
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# def bert_reward(sentence_a, sentence_b):
#     """
#     计算两个句子的相似度（余弦相似度）。
#     """
#     embeddings_a = model.encode(sentence_a, convert_to_tensor=True).cpu()
#     embeddings_b = model.encode(sentence_b, convert_to_tensor=True).cpu()
#     similarity = 1 - cosine(embeddings_a, embeddings_b)
#     return similarity

def call_llm(
        lm_config,
        prompt,
        port = 8000
) -> str:
    response: str
    # port = os.getenv("PORT", 8000)  # 获取环境变量 PORT，默认 8000
    if lm_config == None:
        lm_config = {
            "provider": "huggingface",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "gen_config": {
                "temperature": 0.0,
                "max_new_tokens": 4096
            }
        }
        # import SimpleNamespace
        from types import SimpleNamespace
        lm_config = SimpleNamespace(**lm_config)
    if lm_config.provider == "huggingface":
        assert isinstance(prompt, str)
        error_times = 0
        while error_times < 10:
            try:
                # 直接实现 call_refModel 的逻辑
                url = f"http://localhost:{port}/v1/chat/completions"
                data = {
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": lm_config.gen_config.get("temperature", 0.0),
                    "max_tokens": lm_config.gen_config.get("max_new_tokens", 4096),
                }

                # 如果 lm_config 中有 top_p，则添加到请求中
                if "top_p" in lm_config.gen_config:
                    data["top_p"] = lm_config.gen_config["top_p"]

                # 发送请求
                api_response = requests.post(url, json=data)
                api_response.raise_for_status()
                json_data = api_response.json()
                response = json_data["choices"][0]["message"]["content"]
                break
            except Exception as e:
                print(f"Error: {e}")
                error_times += 1
                print(f"Retrying ({error_times}/10)...")
                time.sleep(10)
    else:
        raise NotImplementedError(
            f"Provider {lm_config.provider} not implemented"
        )
    return response

if __name__ == "__main__":
    input_text = """<|im_start|>system
Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.
The current web page's URL: This is the page you're currently navigating.
The open tabs: These are the tabs you have open.
The previous action: This is the action you just performed. It may be helpful to track your progress.

The actions you can perform fall into several categories:

Page Operation Actions:
`click [id]`: This action clicks on an element with a specific id on the webpage.
`type [id] [content] [press_enter_after=0|1]`: Use this to type the content into the field with id. By default, the ""Enter"" key is pressed after typing unless press_enter_after is set to 0.
`hover [id]`: Hover over an element with id.
`press [key_comb]`:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
`scroll [direction=down|up]`: Scroll the page up or down.

Tab Management Actions:
`new_tab`: Open a new, empty browser tab.
`tab_focus [tab_index]`: Switch the browser's focus to a specific tab using its index.
`close_tab`: Close the currently active tab.

URL Navigation Actions:
`goto [url]`: Navigate to a specific URL.
`go_back`: Navigate to the previously viewed page.
`go_forward`: Navigate to the next page (if a previous 'go_back' action was performed).

Completion Action:
`stop [answer]`: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as ""N/A"" in the bracket.

To be successful, it is very important to follow the following rules:
1. You should only issue an action that is valid given the current observation.
2. You should only issue one action at a time.
3. You should follow the examples to reason step by step and then issue the next action.
4. All reasoning must be inside `<think></think>` tags, and there must be no output before `<think></think>`.
5. After `<think></think>`, only the action should be generated in the correct format, enclosed in code fences. For example:
   <think>This button looks relevant to my goal. Clicking it should take me to the next step.</think>
   ```click [1234]```
6. Issue the stop action when you think you have achieved the objective. Don’t generate anything after stop.
<|im_end|>
7. Always format actions correctly: 
```command [parameters]```
For example, if searching for ""death row inmates in the US"" in a search field with ID `21`, correctly format it as:
```type [21] [death row inmates in the US] [1]```
Avoid incorrect formats that omit brackets around parameters or numeric values.

<|im_start|>user
Objective: total number of death row inmates in the us

URL: http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing
Observation:
[1] RootWebArea 'User:The other Kiwix guy/Landing' focused: True
        [21] textbox ""Search 'Wikipedia'"" required: False
        [23] link 'Go to welcome page'
                [30] button '🏠'
        [24] link ""Go to the main page of 'Wikipedia'""
                [32] button 'Wikipedia'
        [25] link 'Go to a randomly selected page'
                [34] button '🎲'
        [82] StaticText 'Welcome to '
        [83] link 'Wikipedia'
        [84] StaticText 'The free encyclopedia.'
        [371] StaticText '6,489,052'
        [86] StaticText ' articles in '
        [369] link 'English'
        [53] heading 'Arts'
        [89] link 'Architecture'
        [91] link 'Books'
        [93] link 'Cinematography'
        [95] link 'Dance'
        [97] link 'Design'
        [99] link 'Fashion'
        [101] link 'Films'
        [103] link 'Gastronomy'
        [105] link 'Literature'
        [107] link 'Magic (illusion)'
        [109] link 'Music'
        [111] link 'Painting'
        [113] link 'Photography'
        [115] link 'Poetry'
        [117] link 'Sculpture'
        [119] link 'Theatre'
        [55] heading 'Geography'
        [122] link 'Africa'
        [124] link 'Antarctica'
        [126] link 'Arctic'
        [128] link 'Asia'
        [130] link 'Caribbean'
        [132] link 'Central America'
        [134] link 'Europe'
        [136] link 'Latin America'
        [138] link 'Mediterranean'
        [140] link 'Middle East'
        [142] link 'North America'
        [144] link 'Oceania'
        [146] link 'South America'
        [148] link 'Cartography'
        [57] heading 'History'
        [150] link 'Ancient Egypt'
        [152] link 'Ancient Greece'
        [154] link 'Ancient Near East'
        [156] link 'Ancient Rome'
        [158] link 'Archaeology'
        [160] link 'British Empire'
        [162] link 'Byzantine Empire'
        [164] link 'Colonialism'
        [166] link 'Crusades'
        [168] link 'Heraldry'
        [170] link 'History of science'
        [172] link 'Imperial China'
        [174] link 'Indian independence movement'
        [176] link 'Japan'
        [178] link 'Middle Ages'
        [180] link 'Mughal Empire'
        [182] link 'Ottoman Empire'
        [184] link 'Russian Empire'
        [186] link 'Sasanian Empire'
        [188] link 'Seljuk Empire'
        [190] link 'Soviet Union'
        [192] link 'War'
        [59] heading 'Sciences'
Parsed Previous Action:
None
<|im_end|>

<|im_start|>assistant"""
    import os
    # port = os.getenv("PORT", 8000)
    port = 36665
    print(call_llm(lm_config=None, prompt=input_text, port=port))
