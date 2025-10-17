#!/bin/bash
set -x
# 1. begin ray server
host=0.0.0.0
port=30816
# port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
# source server/bin/activate
# python -m verl_tool.servers.ray_serve --host $host --port $port --tool_type "python_code" 2>&1 > /dev/null &
# server_pid=$!
# echo "Server (pid=$server_pid) started at $tool_server_url"

# 2. start api service
model_path="/home/yutao/model/qwen2.5-7b-0816/qwen2.5-7b-2000q-2000q-1369q-nomemory-newbs-old-click-2ep-lr1e-6/checkpoint-636"
max_turns=5
api_host="0.0.0.0"
api_port=5002
# action_stop_tokens= '```\n,<browser>,</action>'
#action_stop_tokens='<action>stop'
action_stop_tokens=''
tensor_parallel_size=1
num_models=3 # number of vllm instances; num_models * tensor_parallel_size should be equal to the number of GPUs
# temp file for action tokens as verl cannot pass special strs as params
max_model_len=32768
action_stop_tokens_file=$(mktemp)
echo "$action_stop_tokens" > $action_stop_tokens_file
echo "action_stop_tokens_file=$action_stop_tokens_file"


source /home/yutao/anaconda3/bin/activate yt_agent
CUDA_VISIBLE_DEVICES=4,5,6 python eval_service/app.py \
    --host $api_host \
    --port $api_port \
    --tool-server-url $tool_server_url \
    --model $model_path \
    --max-turns $max_turns \
    --action_stop_tokens $action_stop_tokens_file \
    --tensor-parallel-size $tensor_parallel_size \
    --num-models $num_models \
    --max-model-len $max_model_len \

api_server_pid=$!
echo "API started at $api_host:$api_port"

# 3. kill all server
# pkill -9 -P $server_pid
# kill -9 $kill $server_pid
pkill -9 -P $api_server_pid
kill -9 $kill $api_server_pid


