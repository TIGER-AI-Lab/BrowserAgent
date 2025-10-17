import time
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor

# =====================
#  1) Pydantic 数据模型
# =====================
# class StartRequest(BaseModel):
#     indexes: List[int]
#     keys: List[str]
class StartRequest(BaseModel):
    questions: List[str]
    gts: List[str]
    keys: List[str]

class StartResponse(BaseModel):
    obs: List[str]

class StepRequest(BaseModel):
    query: List[str]
    keys: List[str]

class StepResponse(BaseModel):
    obs: List[str]
    if_done: List[int]  # -1: invalid, 0: not done, 1: done


# =====================
#  2) 初始化 FastAPI + 状态
# =====================
app = FastAPI()
executor = ProcessPoolExecutor(max_workers=1)

# key 状态字典，按需扩展
key_states = {}  # { key: {"valid": True} }


# =====================
#  3) 两个独立的 worker
# =====================
import pandas as pd
from mini_webarena.object_store import ObjectStore
store = ObjectStore(db_path="browser.db")
from mini_webarena.env_worker import WikiQAEnv

def start_worker_function(args: tuple[str, str, str]) -> str:
    key, question, gt = args
    print(f"key: {key} ####### question: {question} ####### gt: {gt}")
    # row = data_df.iloc[index]
    # question = row["extra_info"]["question"]
    # gt = row["extra_info"]["selected_answer"]

    env = WikiQAEnv(question, gt)
    obs = env.render()  # 初次渲染
    state = env.get_state()
    store.add_object(key, state)
    env.close()
    del env

    return obs

def step_worker_function(args: tuple[str, str]) -> tuple[str, int]:
    key, query = args
    print(f"key: {key} ####### query: {query}")
    state = store.get_object(key)
    if state is None:
        return "", -1

    env = WikiQAEnv(state["question"], state["gt"])
    env.load_state(state)
    obs, reward, done, info = env.step(query)
    if done:
        store.delete_object(key)
    else:
        new_state = env.get_state()
        store.add_object(key, new_state)
    env.close()
    del env

    return obs, 1 if done else 0


# =====================
#  4) /start 接口
# =====================
@app.post("/start", response_model=StartResponse)
def start_endpoint(req: StartRequest):
    obs_list = []

    for key in req.keys:
        key_states[key] = {"valid": True}

    tasks = [(req.keys[i], req.questions[i], req.gts[i]) for i in range(len(req.keys))]
    futures = [executor.submit(start_worker_function, task) for task in tasks]

    for future in futures:
        obs_list.append(future.result())

    return StartResponse(obs=obs_list)


# =====================
#  5) /step 接口
# =====================
@app.post("/step", response_model=StepResponse)
def step_endpoint(req: StepRequest):
    obs_list = []
    if_done_list = []

    tasks = [(req.keys[i], req.query[i]) for i in range(len(req.keys))]
    futures = [executor.submit(step_worker_function, task) for task in tasks]

    for future in futures:
        obs, if_done = future.result()
        obs_list.append(obs)
        if_done_list.append(if_done)

    return StepResponse(obs=obs_list, if_done=if_done_list)


# =====================
#  6) 启动服务器（支持多进程）
# =====================
if __name__ == "__main__":
    uvicorn.run("server:app", host="localhost", port=8002, reload=False)
