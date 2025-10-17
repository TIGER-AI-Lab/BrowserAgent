# import nest_asyncio
# nest_asyncio.apply()
# import asyncio
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import numpy.typing as npt
from beartype import beartype
from beartype.door import is_bearable
from gymnasium import Env
from gymnasium.spaces import Box, Text

from playwright.async_api import Page, ViewportSize, CDPSession

# from .browser_env_async import AsyncScriptBrowserEnv
from .browser_actions import Action, execute_action, get_action_space
from .browser_processors import ObservationHandler, ObservationMetadata
from .utils import (
    AccessibilityTree,
    DetachedPage,
    Observation,
    png_bytes_to_numpy,
    StateInfo
)

import base64
from .scripts import *

Trajectory = list[Union[StateInfo, Action]]

@dataclass
class PlaywrightScript:
    function: str  # goto, get_by_role
    destination: str  # https://www.google.com/, combobox
    name: str | None = None  # Search, Avatar 2009
    operation: str | None = None  # click, fill, press
    value: str | None = None  # avatar movie, Enter


def parse_action(action: str) -> PlaywrightScript:
    splitted = action.strip().split(" ")
    assert len(splitted) >= 2
    match splitted[:2]:
        case ["goto", url]:
            assert len(splitted) == 2
            return PlaywrightScript("goto", url)
        case ["get_by_role", destination]:
            assert len(splitted) >= 4
            match splitted[2:]:
                case [name, operation]:
                    return PlaywrightScript(
                        "get_by_role", destination, name, operation
                    )
                case [name, operation, value]:
                    return PlaywrightScript(
                        "get_by_role", destination, name, operation, value
                    )
                case _:
                    raise ValueError("Invalid action")
        case _:
            raise ValueError(f"Invalid action {action}")

from playwright.sync_api import (
    CDPSession,
    Page,
    Playwright,
    ViewportSize,
    expect,
    sync_playwright,
)

class ScriptBrowserEnv(Env[dict[str, Observation], Action]):
    """
    The goal of this environment is to produce a prototype of a browser environment.
    In the end, we want to support a fully configurable browser environment with wide
    range of action spaces and observation spaces, both structured and unstructured.
    But in this prototype, we just support action space specified by Playwright script,
    and observation space is the html content of the page.
    """

    @beartype
    def __init__(
            self,
            max_page_length: int = 8192,
            headless: bool = True,
            slow_mo: int = 0,
            observation_type: str = "html",
            current_viewport_only: bool = False,
            viewport_size: ViewportSize = {"width": 1280, "height": 720},
            save_trace_enabled: bool = False,
            sleep_after_execution: float = 0.0,
            simple_mode: bool = False,
            page_load_timeout: float = 30.0  # 新增：页面加载超时时间
    ):
        # TODO: make Space[Action] = ActionSpace
        self.action_space = get_action_space()  # type: ignore[assignment]
        self.headless = headless
        self.slow_mo = slow_mo
        self.current_viewport_only = current_viewport_only
        self.reset_finished = False
        self.viewport_size = viewport_size
        self.save_trace_enabled = save_trace_enabled
        self.sleep_after_execution = sleep_after_execution
        self.simple_mode = simple_mode
        self.page_load_timeout = page_load_timeout

        match observation_type:
            case "html" | "accessibility_tree":
                self.text_observation_type = observation_type
                self.image_observation_type = ""
                self.main_observation_type = "text"
            case "image":
                self.image_observation_type = observation_type
                self.text_observation_type = ""  # type: ignore[assignment]
                self.main_observation_type = "image"
            case _:
                raise ValueError(
                    f"Unsupported observation type: {observation_type}"
                )

        self.observation_handler = ObservationHandler(
            self.main_observation_type,
            self.text_observation_type,
            self.image_observation_type,
            self.current_viewport_only,
            self.viewport_size,
            simple_mode = simple_mode
        )

        self.observation_space = (
            self.observation_handler.get_observation_space()
        )

    @beartype
    def setup(self, config_file: Path | None = None) -> None:
        self.context_manager = sync_playwright()
        self.playwright = self.context_manager.start()
        self.browser = self.playwright.chromium.launch(
            headless=self.headless, slow_mo=self.slow_mo,
            args=["--no-sandbox"]
        )

        if config_file:
            with open(config_file, "r") as f:
                instance_config = json.load(f)
        else:
            instance_config = {}

        storage_state = instance_config.get("storage_state", None)
        start_url = instance_config.get("start_url", None)
        geolocation = instance_config.get("geolocation", None)

        self.context = self.browser.new_context(
            viewport=self.viewport_size,
            storage_state=storage_state,
            geolocation=geolocation,
            device_scale_factor=1,
        )
        if self.save_trace_enabled:
            self.context.tracing.start(screenshots=True, snapshots=True)
        if start_url:
            start_urls = start_url.split(" |AND| ")
            for url in start_urls:
                page = self.context.new_page()
                client = page.context.new_cdp_session(
                    page
                )  # talk to chrome devtools
                if self.text_observation_type == "accessibility_tree":
                    client.send("Accessibility.enable")
                page.client = client  # type: ignore # TODO[shuyanzh], fix this hackey client
                self._navigate_with_retry(page, url)
            # set the first page as the current page
            self.page = self.context.pages[0]
            self.page.bring_to_front()
        else:
            self.page = self.context.new_page()
            client = self.page.context.new_cdp_session(self.page)
            if self.text_observation_type == "accessibility_tree":
                client.send("Accessibility.enable")
            self.page.client = client  # type: ignore

    def get_page_client(self, page: Page) -> CDPSession:
        return page.client  # type: ignore

    def _navigate_with_retry(self, page: Page, url: str, max_retries: int = 3, timeout: int = 300000) -> bool:
        """
        带重试机制的页面导航
        Args:
            page: Playwright页面对象
            url: 要访问的URL
            max_retries: 最大重试次数，默认3次
            timeout: 超时时间（毫秒），默认5分钟
        Returns:
            bool: 是否成功导航
        """
        for attempt in range(max_retries):
            try:
                print(f"[DEBUG] 尝试导航到 {url} (第{attempt + 1}/{max_retries}次)")
                page.goto(url, timeout=timeout)
                print(f"[DEBUG] 成功导航到 {url}")
                return True
            except Exception as e:
                print(f"[WARN] 导航失败 (第{attempt + 1}/{max_retries}次): {e}")
                if attempt < max_retries - 1:
                    print(f"[DEBUG] 等待5秒后重试...")
                    time.sleep(5)
                else:
                    print(f"[ERROR] 经过{max_retries}次尝试后仍无法导航到 {url}")
                    raise e
        return False

    def _wait_for_page_ready(self) -> None:
        """等待页面加载完成，确保页面不再处于busy状态"""
        max_wait_time = self.page_load_timeout  # 使用配置的超时时间
        check_interval = 0.2  # 每200ms检查一次，减少频繁检查
        start_time = time.time()
        
        # 检查是否是搜索页面，搜索页面可能需要更长等待时间
        current_url = self.page.url
        is_search_page = 'search' in current_url.lower() or 'query' in current_url.lower()
        if is_search_page:
            max_wait_time = max(max_wait_time, 45.0)  # 搜索页面至少等待45秒
            print(f"[DEBUG] 检测到搜索页面，延长等待时间到 {max_wait_time} 秒")
        
        print(f"[DEBUG] 开始等待页面加载完成... (URL: {current_url})")
        
        try:
            # 第一步：等待基本加载状态
            try:
                self.page.wait_for_load_state('domcontentloaded', timeout=5000)
                print(f"[DEBUG] DOM内容加载完成")
            except Exception as e:
                print(f"[DEBUG] DOM加载等待超时: {e}")
            
            # 第二步：等待网络空闲
            try:
                self.page.wait_for_load_state('networkidle', timeout=3000)
                print(f"[DEBUG] 网络空闲状态达到")
            except Exception as e:
                print(f"[DEBUG] 网络空闲等待超时: {e}")
            
            # 第三步：检查busy状态并等待
            busy_check_count = 0
            max_busy_checks = int(max_wait_time / check_interval)
            
            while time.time() - start_time < max_wait_time and busy_check_count < max_busy_checks:
                try:
                    # 获取accessibility tree来检查是否还在busy状态
                    if self.text_observation_type == "accessibility_tree":
                        client = self.get_page_client(self.page)
                        tree_response = client.send("Accessibility.getFullAXTree")
                        if tree_response and 'nodes' in tree_response:
                            root_node = tree_response['nodes'][0] if tree_response['nodes'] else {}
                            is_busy = root_node.get('busy', False)
                            
                            if not is_busy:
                                print(f"[DEBUG] 页面不再忙碌，等待完成 (耗时: {time.time() - start_time:.2f}秒)")
                                break
                            else:
                                print(f"[DEBUG] 页面仍在忙碌状态，继续等待... ({busy_check_count}/{max_busy_checks})")
                                time.sleep(check_interval)
                                busy_check_count += 1
                        else:
                            print(f"[DEBUG] 无法获取accessibility tree，停止busy检查")
                            break
                    else:
                        # 如果不是accessibility_tree模式，直接跳出
                        break
                        
                except Exception as e:
                    print(f"[DEBUG] busy状态检查失败: {e}")
                    time.sleep(check_interval)
                    busy_check_count += 1
                    continue
            
            # 如果仍然busy，给出警告但继续执行
            if busy_check_count >= max_busy_checks:
                print(f"[WARN] 页面在{max_wait_time}秒后仍处于busy状态，强制继续")
                # 对于持续busy的页面，额外等待更长时间
                time.sleep(2.0)
            else:
                # 额外等待一小段时间确保稳定
                time.sleep(0.3)
                    
        except Exception as e:
            # 如果等待过程中出现异常，记录但不中断流程
            print(f"[WARN] Page ready check failed: {e}")
            # 至少等待一个基本的时间
            time.sleep(1.0)

    def _get_obs(self) -> dict[str, Observation]:
        obs = self.observation_handler.get_observation(
            self.page, self.get_page_client(self.page)
        )
        return obs

    def _get_obs_metadata(self) -> dict[str, ObservationMetadata]:
        metadata = self.observation_handler.get_observation_metadata()
        return metadata

    @beartype
    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, str] | None = None,
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        """
        Reset the environment.
        :param options: options for the environment. The current supported options are:
            - "storage_state": the storage state of the browser. It is a file path to a json file.
        """
        super().reset(seed=seed, options=options)
        if self.reset_finished:
            self.context_manager.__exit__()

        if options is not None and "config_file" in options:
            config_file = Path(options["config_file"])
            if config_file.exists():
                self.setup(config_file=config_file)
            else:
                raise ValueError(f"Config file {config_file} does not exist.")
        else:
            self.setup()
        self.reset_finished = True

        # 等待页面初始加载完成
        self._wait_for_page_ready()

        if self.sleep_after_execution > 0:
            time.sleep(self.sleep_after_execution)

        images = self.modify_page()

        observation = self._get_obs()
        observation_metadata = self._get_obs_metadata()
        info = {
            "page": DetachedPage(self.page.url, ""),
            "fail_error": "",
            "observation_metadata": observation_metadata,
            "images": images,
        }

        return (observation, info)

    @beartype
    def reset_without_config(
            self,
            storage_state: str | None = None,
            start_url: str | None = None,
            geolocation: dict | None = None,
    ) -> tuple[dict, dict]:
        """
        Reset the environment without using an external config file.
        """
        # print("[DEBUG] Starting reset_without_config with storage_state=", storage_state, ", start_url=", start_url,
        #       ", geolocation=", geolocation)

        super().reset()
        if self.reset_finished:
            # print("[DEBUG] Exiting previous browser context.")
            self.context_manager.__exit__()

        # print("[DEBUG] Initializing Context Manager.")
        self.context_manager = sync_playwright()
        # print("[DEBUG] Initializing Playwright.")
        self.playwright = self.context_manager.start()

        # print("[DEBUG] Launching browser with headless=", self.headless, ", slow_mo=", self.slow_mo)
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
            args=["--no-sandbox"]
        )

        # print("[DEBUG] Creating new browser context with viewport=", self.viewport_size, ", storage_state=",
        #       storage_state, ", geolocation=", geolocation)
        self.context = self.browser.new_context(
            viewport=self.viewport_size,
            storage_state=storage_state,
            geolocation=geolocation,
            device_scale_factor=1,
        )

        if self.save_trace_enabled:
            # print("[DEBUG] Enabling tracing.")
            self.context.tracing.start(screenshots=True, snapshots=True)

        if start_url:
            # print("[DEBUG] Opening start URLs:", start_url)
            start_urls = start_url.split(" |AND| ")
            for url in start_urls:
                # print("[DEBUG] Opening URL:", url)
                page = self.context.new_page()
                client = page.context.new_cdp_session(page)
                if self.text_observation_type == "accessibility_tree":
                    # print("[DEBUG] Enabling accessibility tree.")
                    client.send("Accessibility.enable")
                page.client = client  # type: ignore
                self._navigate_with_retry(page, url)
            self.page = self.context.pages[0]
            self.page.bring_to_front()
        else:
            # print("[DEBUG] No start URL provided. Opening a new blank page.")
            self.page = self.context.new_page()
            client = self.page.context.new_cdp_session(self.page)
            if self.text_observation_type == "accessibility_tree":
                # print("[DEBUG] Enabling accessibility tree.")
                client.send("Accessibility.enable")
            self.page.client = client  # type: ignore

        self.reset_finished = True

        # 等待页面初始加载完成
        self._wait_for_page_ready()

        if self.sleep_after_execution > 0:
            # print("[DEBUG] Sleeping for", self.sleep_after_execution, "seconds after execution.")
            time.sleep(self.sleep_after_execution)

        # print("[DEBUG] Modifying page for observation.")
        images = self.modify_page()
        observation = self._get_obs()
        observation_metadata = self._get_obs_metadata()

        info = {
            "page": DetachedPage(self.page.url, ""),
            "fail_error": "",
            "observation_metadata": observation_metadata,
            "images": images,
        }

        # print("[DEBUG] Reset complete. Returning observation and info.")
        return observation, info

    def save_trace(self, trace_path: str | Path) -> None:
        if self.save_trace_enabled:
            self.context.tracing.stop(path=trace_path)

    def close(self) -> None:
        if self.reset_finished:
            self.context_manager.__exit__()

    def step(
            self, action: Action
    ) -> tuple[dict[str, Observation], float, bool, bool, dict[str, Any]]:
        if not self.reset_finished:
            raise RuntimeError("Call reset first before calling step.")
        # print("Now is in env step function")
        success = False
        fail_error = ""
        try:
            self.page = execute_action(
                action,
                self.page,
                self.context,
                self.observation_handler.action_processor,
            )
            success = True
        except Exception as e:
            fail_error = str(e)
            raise e

        # 等待页面加载完成
        self._wait_for_page_ready()

        # hard sleep TODO[shuyanzh] suboptimal, may need to check network
        if self.sleep_after_execution > 0:
            time.sleep(self.sleep_after_execution)

        images = self.modify_page()

        observation = self._get_obs()
        observation_metadata = self._get_obs_metadata()

        info = {
            "page": DetachedPage(self.page.url, self.page.content()),
            "fail_error": fail_error,
            "observation_metadata": observation_metadata,
            "images": images,
        }

        msg = (
            observation,
            float(success),  # reward
            False,  # terminated
            False,  # truncated
            info,
        )
        return msg

    def modify_page(self):
        if self.simple_mode:
            return {}

        self.page.wait_for_timeout(500)
        try:
            self.page.evaluate(remove_id_script)
        except:
            pass

        img_bytes = self.page.screenshot(path="output/screenshot_raw.png")
        raw_image = base64.b64encode(img_bytes).decode()

        self.page.evaluate(mix_marker_script)
        self.page.wait_for_timeout(100)

        # get all clickable elements
        start_id = 0
        elem_items, start_id = self.page.evaluate(get_rect_script, {
            "selector": ".possible-clickable-element",
            "startIndex": start_id
        })

        # get ocr items
        ocr_items = []
        # ocr_items = page.evaluate(canva_handler_script)
        # svg_items, _ = page.evaluate(get_rect_script, {"selector": "svg", "startIndex": -1})
        # ocr_items = ocr_items + svg_items
        # ocr_items, start_id = get_canva_images(ocr_items, img_bytes, start_id)

        items = elem_items + ocr_items

        # mark our own labels and get the images
        items = self.page.evaluate(label_marker_script, items)
        img_bytes = self.page.screenshot(path="output/marked.png")
        marked_image = base64.b64encode(img_bytes).decode()

        self.page.evaluate(remove_label_mark_script)

        return {
            "raw_image": raw_image,
            "marked_image": marked_image,
        }