## Tool Server Usage
We provide a tool server starting command to start any tool server that is supported by verl-tool (see full list in [verl_tool/servers/tools](verl_tool/servers/tools)). To start the tool server, you can use the following command:
```bash
# Start the tool server
host=localhost
port=5500
tool_type=python_code # separate by comma if you want to start multiple tool servers. 
workers_per_tool=4 # number of workers for the tool server, meaning how many threads will be used to handle a single tool request with multiple trajectories
python -m verl_tool.servers.serve --host $host --port $port --tool_type $tool_type --workers_per_tool $workers_per_tool & # run in background
```
After running, you should see the following output. Those marked with 🟢 are active tools, while those marked with ⚪ are inactive tools. `finish` as a tool will always be added to manage the end of each trajectory (e.g. delete env)
```
2025-06-06 02:19:04,764 - __main__ - INFO - Initializing tools: ('python_code',)
2025-06-06 02:19:04,772 - __main__ - INFO - Initialized tool: python_code
2025-06-06 02:19:04,772 - __main__ - INFO - Available Tools:
2025-06-06 02:19:04,773 - __main__ - INFO -   - base: inactive ⚪
2025-06-06 02:19:04,773 - __main__ - INFO -   - text_browser: inactive ⚪
2025-06-06 02:19:04,773 - __main__ - INFO -   - finish: active 🟢
2025-06-06 02:19:04,773 - __main__ - INFO -   - piston: inactive ⚪
2025-06-06 02:19:04,773 - __main__ - INFO -   - ipython_code: inactive ⚪
2025-06-06 02:19:04,773 - __main__ - INFO -   - python_code: active 🟢
2025-06-06 02:19:04,773 - __main__ - INFO -   - bash_terminal: inactive ⚪
2025-06-06 02:19:04,773 - __main__ - INFO -   - sandbox_fusion: inactive ⚪
2025-06-06 02:19:04,773 - __main__ - INFO -   - python_oj: inactive ⚪
2025-06-06 02:19:04,774 - __main__ - INFO - Starting async server on localhost:5500
2025-06-06 02:19:04,774 - __main__ - INFO - Server configured for up to 128 concurrent requests
INFO:     Started server process [493613]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:5500 (Press CTRL+C to quit)
```
To test the tool server, we provide a list of corresponding test scripts in the `verl_tool/servers/tests` directory. For example, to test the `python_code` tool server, you can run the following command:
```bash
# Test the python_code tool server
python -m verl_tool.servers.tests.test_python_code_tool python --url=http://localhost:$port/get_observation
```

## Avaliable Tools
|Tool          |Type            |
|--------------|----------------|
|[Python Interpreter](https://github.com/TIGER-AI-Lab/verl-tool/blob/main/verl_tool/servers/tools/python_code.py) (recommend)|Code Interpreter|
|[Python OJ](https://github.com/TIGER-AI-Lab/verl-tool/blob/main/verl_tool/servers/tools/python_oj.py)|Python Online Judge with test cases|
|[Piston](https://github.com/TIGER-AI-Lab/verl-tool/blob/main/verl_tool/servers/tools/piston.py) (sandbox)|Code Interpreter|
|[Text Broswer](https://github.com/TIGER-AI-Lab/verl-tool/blob/main/verl_tool/servers/tools/text_browser.py)  |Web Broswer     |
|[Base Terminal](https://github.com/TIGER-AI-Lab/verl-tool/blob/main/verl_tool/servers/tools/bash_terminal.py) | Base Terminal |
|Image Processing (Coming soon) | Image Processing |