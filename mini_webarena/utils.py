"""Config for language models."""

# from data/utils.py
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LMConfig:
    """A config for a language model.

    Attributes:
        provider: The name of the API provider.
        model: The name of the model.
        model_cls: The Python class corresponding to the model, mostly for
             Hugging Face transformers.
        tokenizer_cls: The Python class corresponding to the tokenizer, mostly
            for Hugging Face transformers.
        mode: The mode of the API calls, e.g., "chat" or "generation".
    """

    provider: str
    model: str
    model_cls: type | None = None
    tokenizer_cls: type | None = None
    mode: str | None = None
    gen_config: dict[str, Any] = dataclasses.field(default_factory=dict)
    cuda: str = '0'

def construct_llm_config(model_name, inference_endpoint) -> LMConfig:
    llm_config = LMConfig(
        provider="huggingface",
        model=model_name,
        mode="chat",
        cuda="0", # TODO Check whether is cuda 0, btw if use sglang then not need to use cuda
    )
    llm_config.gen_config["temperature"] = 0.8
    llm_config.gen_config["top_p"] = 0.8
    llm_config.gen_config["max_new_tokens"] = 384
    llm_config.gen_config["stop_sequences"] = [None] if None else None
    llm_config.gen_config["max_obs_length"] = 1920
    llm_config.gen_config["model_endpoint"] = inference_endpoint  # User-provided endpoint
    llm_config.gen_config["max_retry"] = 1
    return llm_config

# from broswer/utils.py
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, TypedDict, Union

import numpy as np
import numpy.typing as npt
from PIL import Image


@dataclass
class DetachedPage:
    url: str
    content: str  # html


def png_bytes_to_numpy(png: bytes) -> npt.NDArray[np.uint8]:
    """Convert png bytes to numpy array

    Example:

    >>> fig = go.Figure(go.Scatter(x=[1], y=[1]))
    >>> plt.imshow(png_bytes_to_numpy(fig.to_image('png')))
    """
    return np.array(Image.open(BytesIO(png)))


class AccessibilityTreeNode(TypedDict):
    nodeId: str
    ignored: bool
    role: dict[str, Any]
    chromeRole: dict[str, Any]
    name: dict[str, Any]
    properties: list[dict[str, Any]]
    childIds: list[str]
    parentId: str
    backendDOMNodeId: str
    frameId: str
    bound: list[float] | None
    union_bound: list[float] | None
    offsetrect_bound: list[float] | None


class DOMNode(TypedDict):
    nodeId: str
    nodeType: str
    nodeName: str
    nodeValue: str
    attributes: str
    backendNodeId: str
    parentId: str
    childIds: list[str]
    cursor: int
    union_bound: list[float] | None


class BrowserConfig(TypedDict):
    win_top_bound: float
    win_left_bound: float
    win_width: float
    win_height: float
    win_right_bound: float
    win_lower_bound: float
    device_pixel_ratio: float


class BrowserInfo(TypedDict):
    DOMTree: dict[str, Any]
    config: BrowserConfig


AccessibilityTree = list[AccessibilityTreeNode]
DOMTree = list[DOMNode]


Observation = str | npt.NDArray[np.uint8]


class StateInfo(TypedDict):
    observation: dict[str, Observation]
    info: Dict[str, Any]
