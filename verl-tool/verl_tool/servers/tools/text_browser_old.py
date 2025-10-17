import ray
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .base import BaseTool, register_tool, registered_tools
from mini_webarena.env_worker import WikiQAEnv

@ray.remote
class WikiEnvActor:
    def __init__(self, question: str, gt: str, url: str = None):
        self.env = WikiQAEnv(question, gt, url=url, prompt_format="last")

    def start_env(self) -> str:
        obs = self.env.render()
        return obs

    def step_env(self, query: str) -> (str, int):
        obs, done, valid = self.env.step(query)
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
        return trajectory_id in self.env_actors

    def load_env(self, trajectory_id: str):
        """Return a live actor or `None` if the trajectory is unknown."""
        return self.env_actors.get(trajectory_id)

    def save_env(self, trajectory_id: str, actor):
        """Register / refresh an actor and update LRU ordering."""
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

    def delete_env(self, trajectory_id):
        """Kill and remove the actor."""
        return
        if trajectory_id in self.env_actors:
            ray.kill(self.env_actors[trajectory_id], no_restart=True)
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
        if action == "" or action is None:  # Tentitively allow empty action, since first obs is needed
            return action, True
        pattern = r"<think>.*?</think>\s*(?:```.*?```|<action>.*?</action>)"
        matched = re.search(pattern, action, re.DOTALL)
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

        # 3) Wait for the Ray RPC to finish (blocks the calling thread only).
        result = ray.get(fut)
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
        while len(self.env_actors) > 512:
            # raise RuntimeError("Too many actors, please reduce the number of concurrent requests.")
            oldest = self.actor_creation_order.pop(0)
            print(f"[INFO] Deleting actor {oldest} due to too many actors.")
            if oldest in self.env_actors:
                ray.kill(self.env_actors[oldest], no_restart=True)
                del self.env_actors[oldest]
            if oldest in self.actor_creation_order:
                self.actor_creation_order.remove(oldest)
            # self.delete_env(oldest)