# source .venv-server/bin/activate

ray stop
ray start --head --dashboard-host=0.0.0.0

host=0.0.0.0
# port=$(shuf -i 30000-31000 -n 1)
port=30810
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host $host --port $port --tool_type "text_browser"
server_pid=$!
echo "Server (pid=$server_pid) started at $tool_server_url"

pkill -P -9 $server_pid
kill -9 $kill $server_pid
