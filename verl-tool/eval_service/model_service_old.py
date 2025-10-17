import time
import uuid
import aiohttp
import requests
import regex as re
import openai
import os
import torch
from vllm import SamplingParams
from typing import Dict, Any, List, Tuple
from config import ModelConfig, ToolConfig
from transformers import AutoTokenizer
import asyncio
import random
import subprocess

# 1) A sanitizer that strips all embedded NULs (and, optionally, any
#    other C0 control characters except common whitespace).
CONTROL_CHAR_RE = re.compile(
    # this matches U+0000 through U+001F, excluding tab(09), LF(0A), CR(0D)
    r'[\x00-\x08\x0B\x0C\x0E-\x1F]'
)

def sanitize_request(obj: Any) -> Any:
    """
    Recursively walk through obj and:
      - For dicts: sanitize each value
      - For lists/tuples: sanitize each element
      - For strings: remove embedded nulls (and other control chars)
      - Leave other types untouched
    """
    if isinstance(obj, dict):
        return {sanitize_request(key): sanitize_request(val) for key, val in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(sanitize_request(item) for item in obj)
    elif isinstance(obj, str):
        # strip NUL (\x00) and other C0 control chars
        return CONTROL_CHAR_RE.sub('', obj)
    else:
        return obj
    
class ModelService:
    """verl-tool model inference service"""
    
    call_tool_first = True  # 新增全局变量，默认True
    
    def __init__(self, model_config: ModelConfig, tool_config: ToolConfig):
        """initialize model service"""
        self.model_config = model_config
        self.tool_config = tool_config
        self.model = None
        self.tokenizer = None
        self.session = None
    
    def call_tool_server(self, trajectory_ids: List[str], actions: List[str], finish: List[bool], extra_fields=None, **kwargs) -> Dict[str, Any]:
        """querying the tool server for the observation and done flag"""
        server_url = self.tool_config.tool_server_url
        # prepare payload
        data = {
            "trajectory_ids": trajectory_ids,
            "actions": actions,
            "finish": finish,
            **kwargs
        }
        if extra_fields is not None:
            data["extra_fields"] = extra_fields
        try:
            data = sanitize_request(data)
            response = requests.post(server_url, json=data)
            response.raise_for_status()
            result = response.json()
            return result
        except Exception as e:
            print(f"Error calling tool server: {str(e)}")
            return {
                "observations": [f"Error calling tool server: {str(e)}" for _ in range(len(trajectory_ids))],
                "dones": [True for _ in range(len(trajectory_ids))],
                "valids": [False for _ in range(len(trajectory_ids))]
            }
    
    async def call_tool_server_async(self, trajectory_ids: List[str], actions: List[str], finish: List[bool], **kwargs: Dict[str, List[Any]]) -> Dict[str, Any]:
        """querying the tool server for the observation and done flag using aiohttp"""
        server_url = self.tool_config.tool_server_url
        # prepare payload
        data = {
            "trajectory_ids": trajectory_ids,
            "actions": actions,
            "finish": finish,
            **kwargs
        }
        
        # Create aiohttp session if it doesn't exist
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
        try:
            data = sanitize_request(data)
            async with self.session.post(server_url, json=data) as response:
                response.raise_for_status()
                result = await response.json()
                return result
        except Exception as e:
            print(f"Error calling tool server: {str(e)}")
            return {
                "observations": [f"Error calling tool server: {str(e)}" for _ in range(len(trajectory_ids))],
                "dones": [True for _ in range(len(trajectory_ids))],
                "valids": [False for _ in range(len(trajectory_ids))]
            }
    
    def post_process_observations(self, observations: List[str]):
        next_obs_ids = self.tokenizer(
            observations,
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids'].to(torch.int64)

        if isinstance(self.tool_config.max_obs_length, int) and next_obs_ids.shape[1] > self.tool_config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            if self.config.truncate_obs_side == 'left':
                next_obs_ids = next_obs_ids[:, -self.config.max_obs_length:]
            elif self.config.truncate_obs_side == 'right':
                next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]
            else:
                raise ValueError(f"Invalid truncate_obs_side: {self.config.truncate_obs_side}")
        
        # Convert to list of strings
        next_obs = self.tokenizer.batch_decode(
            next_obs_ids,
            skip_special_tokens=True,
        )
        return next_obs
        
    def load_model(self):
        """load the model using VLLM backend"""
        print(f"Loading Model using VLLM: {self.model_config.model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model)
        # start a VLLM server using vllm.serve
        vllm_args = [f"--{k.replace('_', '-')}" for k in self.model_config.__dict__.keys() if k not in ["model", "api_key", "num_models", "host", "port"]]
        vllm_args = []
        for k, v in self.model_config.__dict__.items():
            if k not in ["model", "api_key", "num_models", "host", "port"]:
                    vllm_args.append(f"--{k.replace('_', '-')}")
                    if not isinstance(v, bool):
                        vllm_args.append(str(v))
        
        host = "0.0.0.0"
        num_models = self.model_config.num_models
        ports = random.sample(range(8000, 9000), num_models)
        self.vllm_processes = []
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join([str(i) for i in range(torch.cuda.device_count())])).split(",")
        tensor_parallel_size = self.model_config.tensor_parallel_size
        gpu_ids_per_model = [gpu_ids[i:i+tensor_parallel_size] for i in range(0, len(gpu_ids), tensor_parallel_size)]
        assert len(gpu_ids) >= num_models * tensor_parallel_size, f"Not enough GPUs available: {len(gpu_ids)} < {num_models * tensor_parallel_size}"
        for i in range(num_models):
            cmd = [
                "vllm", "serve", self.model_config.model, "--api-key", "token-abc123",
                "--host", host, "--port", str(ports[i]), "--disable-uvicorn-access-log", "--disable-log-stats", "--disable-log-requests"
            ] + vllm_args
            env = os.environ.copy()
            env["VLLM_LOGGING_LEVEL"] = "ERROR"
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids_per_model[i])
            vllm_process = subprocess.Popen(cmd, env=env)
            self.vllm_processes.append(vllm_process)
        self.clients = [
            openai.OpenAI(api_key="token-abc123", base_url=f"http://{host}:{ports[i]}/v1") for i in range(num_models)
        ]
        
        # Wait for the service to start (poll the health endpoint)
        max_retries = 60
        retry_interval = 10
        vllm_model_status = [False for _ in range(num_models)]
        for i in range(max_retries):
            for j in range(num_models):
                if vllm_model_status[j]:
                    continue
                try:
                    response = self.clients[j].models.list()
                    vllm_model_status[j] = True
                    print(f"vLLM instance model-{j} status: {response}")
                except Exception as e:
                    # print(f"vLLM instance model-{j} at {host}:{ports[j]} is not ready yet: {str(e)}")
                    continue
            if all(vllm_model_status):
                print(f"vLLM service started successfully with model: {self.model_config.model}")
                return     
            else:
                time.sleep(retry_interval)
        
        # If we get here, the service failed to start
        print("Failed to start one or more vLLM services. Check vLLM logs.")
        for process in self.vllm_processes:
            stderr = process.stderr.read()
            print(f"vLLM stderr: {stderr}")
            process.terminate()
        
        raise RuntimeError("Failed to start vLLM services")
    
    def send_request(self, prompts: List[str], model:str, sampling_params: dict) -> str:
        # randomly select a client and send the request
        client = random.choice(self.clients)
        
        response = client.completions.create(
            model=model,
            prompt=prompts,
            echo=False,
            stream=False,
            **sampling_params
        )
        return response
    
    def generate_with_tools(self, prompts: List[str], sampling_params: dict, extra_fields=None) -> Tuple[List[str], List[str]]:
        """
        Generate text with tool calls in a multi-turn loop.
        Args:
            prompts: Initial prompts for generation
            sampling_params: Sampling parameters for the model
            extra_fields: 额外传递给tool server的内容（如有）
        Returns:
            Tuple of (full_responses, finish_reasons)
        """
        assert sampling_params.get("n", 1) <= 1, "n > 1 is not supported yet for tool generation"
        contexts = prompts
        final_responses = ["" for _ in range(len(prompts))]
        traj_ids = [str(uuid.uuid4()) for _ in range(len(prompts))]
        active_masks = [True for _ in range(len(prompts))]
        finish_reasons = [None for _ in range(len(prompts))]
        model = self.model_config.model

        # call_tool_first逻辑：如果为True，先用action="" call tool server，获得observation并merge进prompt
        obs_first_obs = None
        if getattr(self, 'call_tool_first', False):
            actions = ["" for _ in range(len(prompts))]
            finish = [False for _ in range(len(prompts))]
            tool_result = self.call_tool_server(traj_ids, actions, finish, extra_fields=extra_fields)
            print(f"=========================[DEBUG] tool_result ============================\n {tool_result}")
            observations = tool_result["observations"]
            # 合并observation到prompt
            contexts = [contexts[i] + str(observations[i]) for i in range(len(contexts))]
            # obs也算response，拼到final_responses里
            for i in range(len(contexts)):
                final_responses[i] += str(observations[i])
            obs_first_obs = observations

        print(f"=========================[DEBUG] traj_ids ============================\n {traj_ids}")
        print(f"=========================[DEBUG] contexts ============================\n  {[c for c in contexts]}")

        # keep trying to generate the response until reached the tool-calling limit
        for action_step in range(self.tool_config.max_turns+1):
            print(f"========================================================================================")
            print(f"=========================[DEBUG] action_step {action_step}  ============================\n")
            print(f"========================================================================================")
            if action_step == self.tool_config.max_turns:
                # last turn, don't stop by action stop tokens
                sampling_params.pop("stop")
            active_traj_ids = [traj_ids[i] for i in range(len(traj_ids)) if active_masks[i]]
            active_contexts = [contexts[i] for i in range(len(contexts)) if active_masks[i]]
            print(f"=========================[DEBUG] active_traj_ids ============================\n {active_traj_ids}")
            print(f"=========================[DEBUG] active_contexts ============================\n  {[c for c in active_contexts]}")
            if len(active_contexts) == 0:
                print("=========================[DEBUG] No active contexts, break loop. ============================")
                break
            print(f"=========================[DEBUG] sampling_params ============================\n  {[sampling_params]}")
            # exit(1)
            outputs = self.send_request(
                active_contexts,
                model,
                sampling_params
            )
            print(f"=========================[DEBUG] outputs ============================\n {outputs}")
            active_responses = [outputs.choices[i].text for i in range(len(outputs.choices))]
            active_finish_reasons = [outputs.choices[i].finish_reason for i in range(len(outputs.choices))]
            print(f"=========================[DEBUG] active_responses ============================\n  {active_responses}")
            print(f"=========================[DEBUG] active_finish_reasons ============================\n  {active_finish_reasons}")
            finishes = []
            for i in range(len(active_contexts)):
                finish = True
                if active_finish_reasons[i] == "stop" and outputs.choices[i].stop_reason is not None:
                    active_responses[i] = active_responses[i] + outputs.choices[i].stop_reason
                    finish = False
                finishes.append(finish)
            print(f"=========================[DEBUG] finishes ============================\n  {finishes}")
            tool_responses = self.call_tool_server(
                active_traj_ids,
                active_responses,
                finishes,
                extra_fields=extra_fields
            )
            observations = self.post_process_observations(tool_responses["observations"])
            dones = tool_responses["dones"]
            valids = tool_responses["valids"]
            print(f"=========================[DEBUG] dones ============================\n  {dones}")
            print(f"=========================[DEBUG] valids ============================\n  {valids}")
            active_idx = 0
            for i in range(len(contexts)):
                if active_masks[i]:
                    # 先拼action
                    contexts[i] += active_responses[active_idx]
                    final_responses[i] += active_responses[active_idx]
                    print(f"=========================[DEBUG] {action_step} sample: +action ============================\n{active_responses[active_idx]}")
                    # 只有本轮没done的才拼observation
                    if not dones[active_idx]:
                        contexts[i] += observations[active_idx]
                        final_responses[i] += observations[active_idx]
                        print(f"=========================[DEBUG] {action_step} sample: +observation ============================\n{observations[active_idx]}")
                    finish_reasons[i] = active_finish_reasons[active_idx]
                    active_masks[i] = not dones[active_idx]
                    active_idx += 1
            # print(f"=========================[DEBUG] active_masks ============================\n  {active_masks}")
            # print(f"=========================[DEBUG] final_responses ============================\n  {final_responses}")
            # Server返回dones=True时立刻break
            if all(dones):
                print("=========================[DEBUG] All dones True, break loop. ============================")
                break
        
        return final_responses, finish_reasons
    
    def generate_response(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """process API request and generate response"""
        print(f"Received request: {body}")
        # exit(1)
        
        if "messages" not in body or not body["messages"]:
            raise ValueError("No messages found in the request.")
        # if not 'user' in [message["role"] for message in body["messages"]]:
        #     raise ValueError("No user message found in the request.")
        
        assert body["model"] == self.model_config.model, f"model mismatch: {body['model']} != {self.model_config.model}"
        prompt = self.tokenizer.apply_chat_template(body['messages'],
                                                    add_generation_prompt=True,
                                                    tokenize=False)
        # 判断是否需要传递extra_fields
        extra_fields = [body] if 'url' in body or 'url' in str(body) else None
        if body.get('n', 1) > 1:
            prompts = [prompt for _ in range(body["n"])]
        else:
            prompts = [prompt]

        sampling_params = {
            "temperature": body.get("temperature", 1.0),
            "max_tokens": body.get("max_tokens", body.get("max_completion_tokens", 512)),
            "top_p": body.get("top_p", 1.0),
            "stop": self.tool_config.action_stop_tokens,
        }
        all_responses, finish_reasons = self.generate_with_tools(prompts, sampling_params, extra_fields=extra_fields)
        prompt_tokens = len(self.tokenizer.encode(prompt))
        completion_tokens = 0
        for response in all_responses:
            completion_tokens += len(self.tokenizer.encode(response))
        total_tokens = prompt_tokens + completion_tokens
        
        # format the response into OpenAI-compliant format
        return {
            "id": f"chatcmpl-{str(uuid.uuid4())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_config.model,
            "choices": [
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": all_responses[i],
                    },
                    "finish_reason": finish_reasons[i]
                } for i in range(len(all_responses))
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            } 
        }
        
    async def generate_response_async(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return await asyncio.get_event_loop().run_in_executor(None, self.generate_response, body)
    
    async def close(self):
        """Close any resources (like HTTP sessions and processes) when shutting down"""
        # Close HTTP session
        if self.session:
            await self.session.close()
            self.session = None
            
        # Terminate all VLLM processes
        for process in self.vllm_processes:
            if process:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    
        self.vllm_processes = []
        self.clients = []
        
    def __del__(self):
        """Destructor to ensure resources are cleaned up"""
        asyncio.run(self.close())
