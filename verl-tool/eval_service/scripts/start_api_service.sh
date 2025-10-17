#!/bin/bash
set -x
# 1. begin ray server
host=0.0.0.0
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.ray_serve --host $host --port $port --tool_type "python_code" 2>&1 > /dev/null &
server_pid=$!
echo "Server (pid=$server_pid) started at $tool_server_url"

# 2. start api service
model_path="GAIR/ToRL-1.5B"
max_turns=1
api_host="0.0.0.0"
api_port=5001
action_stop_tokens='```output'
tensor_parallel_size=1
num_models=2 # number of vllm instances; num_models * tensor_parallel_size should be equal to the number of GPUs
# temp file for action tokens as verl cannot pass special strs as params
action_stop_tokens_file=$(mktemp)
echo "$action_stop_tokens" > $action_stop_tokens_file
echo "action_stop_tokens_file=$action_stop_tokens_file"

python eval_service/app.py \
    --host $api_host \
    --port $api_port \
    --tool-server-url $tool_server_url \
    --model $model_path \
    --max-turns $max_turns \
    --action_stop_tokens $action_stop_tokens_file \
    --tensor-parallel-size $tensor_parallel_size \
    --num-models $num_models \

api_server_pid=$!
echo "API started at $api_host:$api_port"

# 3. kill all server
pkill -9 -P $server_pid
kill -9 $kill $server_pid
pkill -9 -P $api_server_pid
kill -9 $kill $api_server_pid
