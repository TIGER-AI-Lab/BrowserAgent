# qwen_client.py

import requests
import time

def call_hf_single(text_query: str) -> str:
    try:
        response = requests.post(
            "http://localhost:8000/generate",
            json={
                "prompt": text_query,
                "max_new_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9
            },
            timeout=20
        )
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: HTTP {response.status_code}"
    except Exception as e:
        return f"Error: {e}"

def call_sglang_single(
        prompt,
        port = 36182
) -> str:
    response: str
    # port = os.getenv("PORT", 8000)  # 获取环境变量 PORT，默认 8000
    lm_config = {
        "provider": "huggingface",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "gen_config": {
            "temperature": 0.0,
            "max_new_tokens": 4096
        }
    }
    from types import SimpleNamespace
    lm_config = SimpleNamespace(**lm_config)
    if lm_config.provider == "huggingface":
        assert isinstance(prompt, str)
        error_times = 0
        while error_times < 10:
            try:
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

def call_gemini_single(prompt, port = 36182) -> str:
    url = f"http://127.0.0.1:{port}/gemini_inference"
    try:
        resp = requests.post(url, json={"prompt": prompt})
        if resp.status_code != 200:
            raise RuntimeError(f"Server returned status {resp.status_code}, body: {resp.text}")
        response_data = resp.json()
        result_str = response_data.get("result", "")
        if result_str == "Error":
            raise RuntimeError("Error in Gemini server inference.")
        # print("#"*100)
        # print(prompt)
        # print("%"*100)
        # print(result_str)
        # print("#"*100)
        return result_str
    except Exception as e:
        raise RuntimeError(f"Failed to call gemini_inference: {e}")

def call_gpt4o_single(text_query: str) -> str:
    from openai_api import gpt4o_wrapper
    try:
        return gpt4o_wrapper.call_gpt4o_single(text_query)
    except Exception as e:
        print(f"Error: {e}")
        return ""


def call_sglang_local(
        prompt,
        port = 36182
) -> str:
    pass

if __name__ == "__main__":
    prompt = "用一句话介绍一下你自己。"
    result = call_gemini_single(prompt)
    print(f"Prompt: {prompt}\nResponse: {result}")
