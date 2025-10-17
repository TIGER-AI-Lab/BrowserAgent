import ray
import re
import asyncio
import time
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .base import BaseTool, register_tool, registered_tools
from mini_webarena.env_worker import WikiQAEnv

@ray.remote
class WikiEnvActor:
    def __init__(self, question: str, gt: str, url: str = None):
        print(f"[DEBUG] WikiEnvActor.__init__ - question: {question[:50]}..., gt: {gt[:50]}..., url: {url}")
        # self.env = WikiQAEnv(question, gt, url=url, prompt_format="last")
        # --- robust env creation with minimal retry ---
        max_retries = 3
        backoff_sec = 10
        for attempt in range(1, max_retries + 1):
            try:
                self.env = WikiQAEnv(question, gt, url=url, prompt_format="last")
                break  # success
            except Exception as e:
                print(f"[WARN] WikiEnvActor init attempt {attempt}/{max_retries} failed: {e}")
                if attempt == max_retries:
                    raise
                time.sleep(backoff_sec)
        
        # 添加空闲 TTL watchdog（类似 sandbox_r2e）
        self._ttl_seconds = 1200  # 20 分钟
        self._last_access = time.time()
        
        def _watchdog():
            """每 5 分钟检查一次；若超时则自杀退出 Actor。"""
            while True:
                time.sleep(300)
                if time.time() - self._last_access > self._ttl_seconds:
                    print(f"[WikiEnvActor] idle for >{self._ttl_seconds}s, exiting.")
                    try:
                        self.env.close()
                        ray.actor.exit_actor()
                    except Exception:
                        import os
                        os._exit(0)
        
        threading.Thread(target=_watchdog, daemon=True).start()

    def start_env(self) -> str:
        print(f"[DEBUG] WikiEnvActor.start_env")
        self._last_access = time.time()
        obs = self.env.render()
        return obs

    def step_env(self, query: str) -> (str, int):
        print(f"[DEBUG] WikiEnvActor.step_env - query: {query[:100]}...")
        self._last_access = time.time()
        # obs, done, valid = self.env.step(query)
        max_retries = 3
        backoff_sec = 5
        for attempt in range(1, max_retries + 1):
            try:
                obs, done, valid = self.env.step(query)
                break
            except Exception as e:
                print(f"[WARN] WikiEnvActor.step_env retry {attempt}/{max_retries} due to: {e}")
                if attempt == max_retries:
                    raise
                time.sleep(backoff_sec)
        if done:
            self.env.close()
        return obs, done, valid


@register_tool
class TextBrowserTool(BaseTool):
    """
    TextBrowserTool uses Ray actors to manage WikiQAEnv sessions.
    Each trajectory_id has a dedicated actor. It supports initial
    render (action=None) and step operations.
    """
    tool_type = "text_browser"

    def __init__(self, num_workers=32):
        print(f"[DEBUG] TextBrowserTool.__init__ - num_workers: {num_workers}")
        super().__init__(num_workers)
        # Maps trajectory_id to Ray Actor
        self.env_actors = {}
        # Track creation order for cleanup
        self.actor_creation_order = []

    # -------------------------------------------------------------------------
    # BaseTool interface methods (some are no-ops here, but we must implement them)
    # -------------------------------------------------------------------------
    def get_usage_inst(self) -> str:
        """Return usage instructions."""
        return "TextBrowserTool uses Ray actors to manage WikiQAEnv sessions."

    def has_env(self, trajectory_id):
        print(f"[DEBUG] TextBrowserTool.has_env - trajectory_id: {trajectory_id}")
        return trajectory_id in self.env_actors

    def load_env(self, trajectory_id: str):
        """Return a live actor or `None` if the trajectory is unknown."""
        print(f"[DEBUG] TextBrowserTool.load_env - trajectory_id: {trajectory_id}")
        return self.env_actors.get(trajectory_id)

    def save_env(self, trajectory_id: str, actor):
        """Register / refresh an actor and update LRU ordering."""
        print(f"[DEBUG] TextBrowserTool.save_env - trajectory_id: {trajectory_id}, actor: {actor}")
        # Should not exist if exist;
        if self.env_actors.get(trajectory_id) is None:
            self.env_actors[trajectory_id] = actor
            self.actor_creation_order.append(trajectory_id)
            self._cleanup_actors_if_needed()
        else:
            # If it exists, check if it's the same actor, otherwise raise an error
            if self.env_actors[trajectory_id] != actor:
                raise RuntimeError(f"Actor with trajectory_id {trajectory_id} already exists.")
            if trajectory_id in self.actor_creation_order:
                self.actor_creation_order.remove(trajectory_id)
            self.actor_creation_order.append(trajectory_id)

    async def asave_env(self, trajectory_id: str, actor):
        """Async version of save_env"""
        print(f"[DEBUG] TextBrowserTool.asave_env - trajectory_id: {trajectory_id}, actor: {actor}")
        if self.env_actors.get(trajectory_id) is None:
            self.env_actors[trajectory_id] = actor
            self.actor_creation_order.append(trajectory_id)
            await self._acleanup_actors_if_needed()
        else:
            if self.env_actors[trajectory_id] != actor:
                raise RuntimeError(f"Actor with trajectory_id {trajectory_id} already exists.")
            if trajectory_id in self.actor_creation_order:
                self.actor_creation_order.remove(trajectory_id)
            self.actor_creation_order.append(trajectory_id)

    def delete_env(self, trajectory_id):
        """同步封装器，兼容旧代码路径"""
        print(f"[DEBUG] TextBrowserTool.delete_env - trajectory_id: {trajectory_id}")
        asyncio.run(self.adelete_env(trajectory_id))
    
    async def adelete_env(self, trajectory_id):
        """异步版本的 delete_env"""
        print(f"[DEBUG] TextBrowserTool.adelete_env - trajectory_id: {trajectory_id}")
        if trajectory_id in self.env_actors:
            actor = self.env_actors[trajectory_id]
            try:
                # 尝试优雅关闭
                await asyncio.sleep(0)  # 让出控制权
            except Exception as e:
                print(f"Error closing env for trajectory_id: {trajectory_id}: {e}")
            
            try:
                ray.kill(actor, no_restart=True)
            except Exception as e:
                print(f"Error killing actor for trajectory_id: {trajectory_id}: {e}")
            1
            del self.env_actors[trajectory_id]
            
        if trajectory_id in self.actor_creation_order:
            self.actor_creation_order.remove(trajectory_id)

    def parse_action(self, action):
        """
        检查action是否包含如下两种格式之一：
        1. <think>.*?</think> 后跟若干空白和换行，再跟 ```.*?```
        2. <think>.*?</think> 后跟若干空白和换行，再跟 <action>.*?</action>
        前后都允许有任意数量空格和回车。
        匹配则返回(action, True)，否则(action, False)
        """
        print(f"[DEBUG] TextBrowserTool.parse_action - action: {action[:100] if action else 'None'}...")
        if action == "" or action is None:  # Tentitively allow empty action, since first obs is needed
            return action, True
        pattern = r"<think>.*?</think>\s*(?:```.*?```|<action>.*?</action>)"
        matched = re.search(pattern, action, re.DOTALL)
        #print("if_matched:",bool(matched))
        return action, bool(matched)

    def conduct_action(self, trajectory_id: str, action: str, extra_field: dict):
        """
        Execute a *single* action on the environment for `trajectory_id`.

        Returns
        -------
        obs : str
            Environment observation (empty string if episode finished).
        done : bool
            Whether the episode ended with this step.
        valid : bool
            Whether the action itself was valid.
        """
        print(f"[DEBUG] TextBrowserTool.conduct_action - trajectory_id: {trajectory_id}, action: {action[:100] if action else 'None'}..., extra_field keys: {list(extra_field.keys())}")
        
        # 1) Ensure an actor exists (lazy creation).
        actor = self.load_env(trajectory_id)
        if actor is None:
            # Create a brand-new WikiEnvActor for this trajectory.
            question = extra_field.get("question", "placeholder")
            gt       = extra_field.get("gt",        "placeholder")
            url      = extra_field.get("url",       None)
            actor = WikiEnvActor.remote(question, gt, url)
            self.save_env(trajectory_id, actor)

        # 2) Decide whether we are rendering the first page or taking a step.
        fut = (
            actor.start_env.remote()
            if action is None or action == ""
            else actor.step_env.remote(action)
        )
        #print(f"[DEBUG] TextBrowserTool.conduct_action_after - trajectory_id: {trajectory_id}, action: {action[:100] if action else 'None'}..., extra_field keys: {list(extra_field.keys())}")

        # 3) Wait for the Ray RPC to finish (blocks the calling thread only).
        result = ray.get(fut)
        print(f"[DEBUG] TextBrowserTool.conduct_action_after - trajectory_id: {trajectory_id}, action: {action[:100] if action else 'None'}..., extra_field keys: {list(extra_field.keys())}")

        if isinstance(result, tuple):           # step_env
            obs, done, valid = result
        else:                                   # start_env
            obs, done, valid = result, False, True

        # 新增：打印输入和输出信息
        output = (
            "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
            f"trajectory_id: {trajectory_id}\n"
            f"action: {action}\n"
            f"extra_field: {extra_field}\n"
            f"observation: {obs}\n"
            "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        )
        print(output)

        # 4) Refresh LRU order *after* the step.
        if trajectory_id in self.actor_creation_order:
            self.actor_creation_order.remove(trajectory_id)
        self.actor_creation_order.append(trajectory_id)

        # 5) Clean-up if the episode finished.
        if done:
            obs = ""            # Clear the final observation
            self.delete_env(trajectory_id)

        if not valid:
            obs = "The action is invalid, please retry"

        return obs, done, valid

    async def aconduct_action(self, trajectory_id: str, action: str, extra_field: dict):
        """异步版本的 conduct_action"""
        print(f"[DEBUG] TextBrowserTool.aconduct_action - trajectory_id: {trajectory_id}, action: {action[:100] if action else 'None'}..., extra_field keys: {list(extra_field.keys())}")
        
        actor = self.load_env(trajectory_id)
        if actor is None:
            # 异步创建 actor
            question = extra_field.get("question", "placeholder")
            gt = extra_field.get("gt", "placeholder")
            url = extra_field.get("url", None)
            actor = WikiEnvActor.remote(question, gt, url)
            await self.asave_env(trajectory_id, actor)
        
        # 将 Ray 调用异步化
        obj_ref = (
            actor.start_env.remote()
            if action is None or action == ""
            else actor.step_env.remote(action)
        )
        
        try:
            # 等待 Ray 对象引用，带超时
            result = await asyncio.wait_for(obj_ref, timeout=300)
        except asyncio.TimeoutError:
            return "[TIMEOUT] (aconduct_action)", True, False
        except Exception as e:
            return f"Error: {e}", False, False
        
        # 解包结果
        if isinstance(result, tuple):
            obs, done, valid = result
        else:
            obs, done, valid = result, False, True
        
        # 刷新 LRU 顺序
        if trajectory_id in self.actor_creation_order:
            self.actor_creation_order.remove(trajectory_id)
        self.actor_creation_order.append(trajectory_id)
        
        # 如果完成则清理
        if done:
            obs = ""
            await self.adelete_env(trajectory_id)
        
        if not valid:
            obs = "The action is invalid, please retry"
        
        return obs, done, valid

    async def aget_observations(self, trajectory_ids, actions, extra_fields):
        """使用信号量进行并发控制的异步批处理版本"""
        print(f"[DEBUG] TextBrowserTool.aget_observations - num_trajectories: {len(trajectory_ids)}")
        
        sem = asyncio.Semaphore(self.num_workers)
        
        async def _task(i):
            async with sem:
                try:
                    extra = extra_fields[i].get("extra_fields", extra_fields[i])
                    return i, *await self.aconduct_action(
                        trajectory_ids[i], actions[i], extra
                    ), None
                except Exception as e:
                    # 出错时清理
                    try:
                        await self.adelete_env(trajectory_ids[i])
                    except Exception as cleanup_e:
                        print(f"[ERROR] Cleanup failed for {trajectory_ids[i]}: {cleanup_e}")
                    return i, "", False, False, e
        
        coros = [_task(i) for i in range(len(trajectory_ids))]
        results = await asyncio.gather(*coros, return_exceptions=False)
        
        # 初始化
        n = len(trajectory_ids)
        observations = [""] * n
        dones = [False] * n
        valid_flags = [True] * n
        
        # 处理结果
        for i, o, d, v, err in results:
            observations[i], dones[i], valid_flags[i] = o, d, v
            if err:
                print(f"[ERROR] trajectory_id={trajectory_ids[i]}: {err}")
        
        # 保持现有的日志记录逻辑
        try:
            import json
            from pathlib import Path
            log_path = Path("browser_server_logs.jsonl")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "input": {
                                "trajectory_ids": trajectory_ids,
                                "actions": actions,
                                "extra_fields": extra_fields,
                            },
                            "output": {
                                "observations": observations,
                                "dones": dones,
                                "valid_flags": valid_flags,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as e:
            print(f"[WARN] Failed to write browser_server_logs.jsonl: {e}")
        
        return observations, dones, valid_flags

    def get_observations(self, trajectory_ids, actions, extra_fields):
        """
        Batched version of `conduct_action` with thread-pool parallelism.
        (A process-pool is **not** required; Ray already runs the envs
        out-of-process.)

        Parameters
        ----------
        trajectory_ids : list[str]
        actions        : list[str | None]
        extra_fields   : list[dict]

        Returns
        -------
        observations : list[str]
        dones        : list[bool]
        valid_flags  : list[bool]
        """
        print(f"[DEBUG] TextBrowserTool.get_observations - num_trajectories: {len(trajectory_ids)}")
        
        # print("[INFO] Using thread pool for parallel processing...")
        # print("[INFO] trajectory_ids:", trajectory_ids)
        # print("[INFO] actions:", actions)
        # print("[INFO] extra_fields:", extra_fields)

        import json
        from pathlib import Path
        from concurrent.futures import ThreadPoolExecutor

        n = len(trajectory_ids)
        observations = [""]   * n
        dones        = [False] * n
        valid_flags  = [True]  * n

        # ----------------------------------------------------------------- #
        # Parallel fan-out using a thread pool                              #
        # ----------------------------------------------------------------- #
        def _worker(idx: int):
            tid   = trajectory_ids[idx]
            act   = actions[idx]
            extra = extra_fields[idx].get("extra_fields", extra_fields[idx])
            try:
                return (*self.conduct_action(tid, act, extra), None)
            except Exception as e:
                return ("", False, False, e)   # bubble error to main thread

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = [pool.submit(_worker, i) for i in range(n)]
            for i, fut in enumerate(futures):
                obs, done, valid, err = fut.result()
                observations[i] = obs
                dones[i]        = done
                valid_flags[i]  = valid
                if err:
                    print(f"[ERROR] trajectory_id={trajectory_ids[i]}: {err}")

        # ----------------------------------------------------------------- #
        # Fire-and-forget JSONL logging                                     #
        # ----------------------------------------------------------------- #
        try:
            log_path = Path("browser_server_logs.jsonl")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "input": {
                                "trajectory_ids": trajectory_ids,
                                "actions": actions,
                                "extra_fields": extra_fields,
                            },
                            "output": {
                                "observations": observations,
                                "dones": dones,
                                "valid_flags": valid_flags,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as e:
            # Logging failures must *never* break main logic
            print(f"[WARN] Failed to write browser_server_logs.jsonl: {e}")

        return observations, dones, valid_flags

    def _cleanup_actors_if_needed(self):
        """Remove oldest actors if count exceeds limit."""
        print(f"[DEBUG] TextBrowserTool._cleanup_actors_if_needed - current_actors: {len(self.env_actors)}")
        while len(self.env_actors) > 512:
            if not self.actor_creation_order:
                break
            oldest = self.actor_creation_order.pop(0)
            print(f"[INFO] Deleting actor {oldest} due to too many actors.")
            try:
                self.delete_env(oldest)
            except Exception as e:
                print(f"[ERROR] Failed to delete actor {oldest}: {e}")
                if oldest in self.env_actors:
                    del self.env_actors[oldest]
    
    async def _acleanup_actors_if_needed(self):
        """异步版本的清理"""
        print(f"[DEBUG] TextBrowserTool._acleanup_actors_if_needed - current_actors: {len(self.env_actors)}")
        while len(self.env_actors) > 512:
            if not self.actor_creation_order:
                break
            oldest = self.actor_creation_order.pop(0)
            print(f"[INFO] Deleting actor {oldest} due to too many actors.")
            try:
                await self.adelete_env(oldest)
            except Exception as e:
                print(f"[ERROR] Failed to delete actor {oldest}: {e}")
                if oldest in self.env_actors:
                    del self.env_actors[oldest]