ray stop
CUDA_VISIBLE_DEVICES=0,1,2,3 ray start --head --dashboard-host=0.0.0.0  --num-cpus=32

set -x
dataset_name=wikiQA
train_data=data/$dataset_name/train.parquet
val_data=data/$dataset_name/test.parquet
# model_name=Qwen/Qwen2.5-3B
# model_name=/home/zhiheng/cogito/base_models/qwen2.5-0.5b-wiki
# model_name=/home/zhiheng/verl/experiments/wiki/qwen2.5-1.5b-direct_plain-hard
model_name=/map-vepfs/zhiheng/base_model/model_3B/qwen2.5-3b-1epoch-hard
rl_alg=grpo # gae(ppo) or grpo, if grpo, then better set n>1 otherwise the group norm can not be effective

n_nodes=1
n=4

#n_gpus_per_node=2
#batch_size=8
#ppo_mini_batch_size=2

n_gpus_per_node=4
batch_size=16
ppo_mini_batch_size=4


max_prompt_length=2048 # bottleneck of the rollout, by default keep the right side
max_response_length=5120 # bottleneck of the right side, will used in the training
max_obs_length=2048 # Not the bottleneck, the obs is much shorter than this
temperature=0.5
strategy="fsdp_agent" # remove _agent for normal verl behavior
valid_actions="[]" # "[answer,python]" are two valid actions, they are used to determine the stop
token of
# each action, which are </answer> and </python> respectively

# === begin, added by Zhiheng ===
max_action_length=512
rolling_with_prompt=False
call_tool_first=True
truncate_obs_side=left # This is weird but required in the current code
truncate_response_side=left
min_action_num=5
mirco_batch_size=1
mirco_batch_size_non_train=2
max_start_length=1536 # System prompt is always length 800+, not the bottleneck
# === end, added by Zhiheng ===

model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name="MAP-3B-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}"
export VERL_RUN_ID=$run_name
export VLLM_ATTENTION_BACKEND=XFORMERS

host=0.0.0.0
port=$(shuf -i 30000-31000 -n 1)
# port=30815
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host $host --port $port --tool_type "text_browser" &
server_pid=$!
echo "Server (pid=$server_pid) started at $tool_server_url"

# actor_rollout_ref.rollout.enforce_eager=False \
# actor_rollout_ref.rollout.free_cache_engine=False \

# export VLLM_USE_V1=1
# actor_rollout_ref.agent.max_turns is for debug only
PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    reward_model.reward_manager=wikiRL \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$mirco_batch_size \
    actor_rollout_ref.actor.strategy=$strategy \
    +actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    +actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    +actor_rollout_ref.agent.max_response_length=$max_response_length \
    +actor_rollout_ref.agent.max_start_length=$max_start_length \
    +actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    +actor_rollout_ref.agent.max_action_length=$max_action_length \
    +actor_rollout_ref.agent.rolling_with_prompt=$rolling_with_prompt \
    +actor_rollout_ref.agent.call_tool_first=call_tool_first \
    +actor_rollout_ref.agent.min_action_num=$min_action_num \
    +actor_rollout_ref.agent.truncate_response_side=$truncate_response_side \
    +actor_rollout_ref.agent.truncate_obs_side=$truncate_obs_side \
    +actor_rollout_ref.agent.max_turns=5 \
    +actor_rollout_ref.agent.num_gpus=$n_gpus_per_node \
    +actor_rollout_ref.agent.valid_actions=$valid_actions \
    +actor_rollout_ref.agent.no_action_as_stop=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$mirco_batch_size_non_train \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$mirco_batch_size_non_train \
    critic.optim.lr=5e-7 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.ppo_micro_batch_size_per_gpu=$mirco_batch_size \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='wikiRL' \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=5


pkill -P -9 $server_pid
kill -9 $kill $server_pid

# loss from 1e-5 to 5e-7;
