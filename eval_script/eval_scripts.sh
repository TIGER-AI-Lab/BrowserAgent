host=0.0.0.0
port=30815
tool_server_url=http://$host:$port/get_observation
PYTHONPATH=/home/zhiheng/cogito/verl-tool python -m verl_tool.servers.serve --host 0.0.0.0 --port 30815 --tool_type "text_browser" &
server_pid=$!
echo "Server (pid=$server_pid) started at $tool_server_url"


python run_eval.py \
  --server_port 30815 \
  --traj_csv_path result/traj_4B-sft-ti.csv \
  --result_csv_path result/result_4B-sft-ti.csv \
  --data_path ./data/dev.parquet \
  --cuda "0" \
  --model_path /home/zhiheng/cogito/base_models/qwen3-4b-dp-hard-ti \
  --obs_type tuncate_input

python run_eval.py \
  --server_port 30815 \
  --traj_csv_path result/traj_4B-sft-dp.csv \
  --result_csv_path result/result_4B-sft-dp.csv \
  --data_path ./data/dev.parquet \
  --cuda "0" \
  --model_path /home/zhiheng/cogito/base_models/qwen3-4b-dp-hard-dp

python run_eval.py \
  --server_port 30815 \
  --traj_csv_path result/traj_3B-rl40.csv \
  --result_csv_path result/result_3B-rl40.csv \
  --data_path ./data/dev.parquet \
  --cuda "1" \
  --model_path /home/zhiheng/cogito/base_models/3B-10K-40Step
  
python run_eval.py \
  --server_port 30815 \
  --traj_csv_path result/traj_3B-rl10.csv \
  --result_csv_path result/result_3B-rl10.csv \
  --data_path ./data/dev.parquet \
  --cuda "2" \
  --model_path /home/zhiheng/cogito/base_models/qwen2.5-3b-baseline-step10

python run_eval.py \
  --server_port 30815 \
  --traj_csv_path result/traj_7B-sft-dp.csv \
  --result_csv_path result/result_7B-sft-dp.csv \
  --data_path ./data/dev.parquet \
  --cuda "3" \
  --model_path /home/zhiheng/cogito/base_models/qwen2.5-7b-dp-hard

python run_eval.py --server_port 30815 --traj_csv_path result/traj_7B-rl30.csv --result_csv_path result/result_7B-rl30.csv --data_path ./data/dev.parquet --cuda "0" --model_path /home/zhiheng/cogito/base_models/qwen_2.5_7b_dp_30step_hf

pkill -P -9 $server_pid
kill -9 $kill $server_pid
