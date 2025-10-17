#!/bin/bash

# === 脚本配置 ===
CMD="python your_script.py"          # 修改为你要运行的命令
RESULT_LOG="result.txt"              # 指令输出保存位置
KEEPER_LOG="keeper.txt"              # 脚本日志输出位置
TIMEOUT_SECONDS=3600                 # 1 小时 = 3600 秒

# === 主逻辑 ===
{
  echo "[`date`] Starting command..."
  
  # 启动命令并保存输出，放后台执行
  $CMD > "$RESULT_LOG" 2>&1 &
  CMD_PID=$!

  echo "[`date`] Command started with PID $CMD_PID. It will be killed after $TIMEOUT_SECONDS seconds."

  # 睡眠指定时间
  sleep "$TIMEOUT_SECONDS"

  # 尝试结束进程
  echo "[`date`] Time's up. Killing process $CMD_PID..."
  kill $CMD_PID 2>/dev/null

  sleep 2

  # 如果没杀掉就强杀
  if kill -0 $CMD_PID 2>/dev/null; then
    echo "[`date`] Process still alive. Sending SIGKILL..."
    kill -9 $CMD_PID 2>/dev/null
  else
    echo "[`date`] Process $CMD_PID terminated."
  fi

  echo "[`date`] Script finished."
} > "$KEEPER_LOG" 2>&1 &

