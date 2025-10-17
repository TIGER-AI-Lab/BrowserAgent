# hf_utils.py
from text_generation import Client  # type: ignore

def generate_from_huggingface_completion(
    prompt: str,
    model_endpoint: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_sequences: list[str] | None = None,
) -> str:
    client = Client(model_endpoint, timeout=60)
    generation: str = client.generate(
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        stop_sequences=stop_sequences,
    ).generated_text

    return generation

# tokenizers.py
from typing import Any
from transformers import AutoTokenizer  # type: ignore

class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        # if provider == "openai":
        #     self.tokenizer = tiktoken.encoding_for_model(model_name)
        # elif provider == "huggingface":
        if provider == "huggingface":
            print(model_name)
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

# utils.py
import argparse
from typing import Any
from transformers import AutoTokenizer, AutoModel

model = None
tokenizer = None

def call_llm(
        lm_config,
        prompt:  str | list[Any] | dict[str, Any],
) -> str:
    response: str
    if lm_config.provider == "huggingface":
        assert isinstance(prompt, str)
        error_times = 0
        while error_times < 10:
            try:
                response = generate_from_huggingface_completion(
                    prompt=prompt,
                    model_endpoint=lm_config.gen_config["model_endpoint"],
                    temperature=lm_config.gen_config["temperature"],
                    top_p=lm_config.gen_config["top_p"],
                    stop_sequences=lm_config.gen_config["stop_sequences"],
                    max_new_tokens=lm_config.gen_config["max_new_tokens"],
                )
                break
            except Exception as e:
                print(e)
                error_times += 1
                print(f"Error times: {error_times}")
                import time
                time.sleep(10)
    else:
        raise NotImplementedError(
            f"Provider {lm_config.provider} not implemented"
        )
    return response


if __name__ == "__main__":
    # 测试参数配置
    prompt = "What is the meaning of life?"
    # 模型服务端点，这里假设你在本地通过 Docker 启动了 GPT-2 模型服务
    model_endpoint = "http://localhost:9982"
    temperature = 0.7
    top_p = 0.9
    max_new_tokens = 50
    stop_sequences = None  # 或者可以设置为例如 ["\n"] 来在遇到换行时停止生成

    # 调用生成函数
    generated_text = generate_from_huggingface_completion(
        prompt,
        model_endpoint,
        temperature,
        top_p,
        max_new_tokens,
        stop_sequences,
    )

    # 打印结果
    print("Prompt:")
    print(prompt)
    print("\nGenerated Text:")
    print(generated_text)
