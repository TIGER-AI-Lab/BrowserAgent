from dataclasses import dataclass

@dataclass
class AgentActorConfig:
    enable_agent: bool=True
    max_turns: int=0
    min_turns: int=0
    max_start_length: int=None
    max_prompt_length: int=None
    max_response_length: int=None
    max_obs_length: int=None
    max_action_length: int=None
    tool_server_url: str = None
    n: int=1
    truncate_obs_side: str='left'
    truncate_response_side: str='left'
    rolling_with_prompt: bool=False
    call_tool_first: bool=False
    action_stop_tokens: list=None
    additional_eos_token_ids: list=None
    mask_observations: bool=True
    force_finish_for_last_turn: bool=False
    debug: bool=False
    enable_mtrl: bool=False
    mtrl_role: str="user"
    mtrl_sep: str= "\n<|im_start|>system\n{obs}<|im_end|>\n<|im_start|>assistant\n"
    turn_end_token: str="<|im_end|>"
