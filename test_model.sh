#!/bin/bash

# VLLM模型测试脚本
# 用于测试部署的模型是否正常运行

# 服务器配置（与deploy_vllm.sh保持一致）
HOST="localhost"
PORT=5004
API_KEY="sk-proj-1234567890"

echo "正在测试VLLM模型服务..."
echo "服务地址: http://$HOST:$PORT"
echo "================================"

# 1. 检查服务器是否响应
echo "1. 检查服务器连接状态..."
if curl -s --connect-timeout 5 http://$HOST:$PORT/health > /dev/null 2>&1; then
    echo "✅ 服务器连接正常"
else
    echo "❌ 服务器连接失败，请检查VLLM服务是否启动"
    exit 1
fi

# 2. 获取模型信息
echo ""
echo "2. 获取模型信息..."
curl -s -X GET "http://$HOST:$PORT/v1/models" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" | jq '.' 2>/dev/null || echo "模型信息获取失败或jq未安装"

# 3. 测试简单对话
echo ""
echo "3. 测试模型对话功能..."
RESPONSE=$(curl -s -X POST "http://$HOST:$PORT/v1/chat/completions" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen2.5-7b",
        "messages": [
            {"role": "user", "content": "你好，请简单介绍一下你自己。"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }')

if echo "$RESPONSE" | grep -q "choices"; then
    echo "✅ 模型对话测试成功"
    echo "模型回复："
    echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null || echo "$RESPONSE"
else
    echo "❌ 模型对话测试失败"
    echo "错误信息: $RESPONSE"
fi

# 4. 测试流式输出
echo ""
echo "4. 测试流式输出功能..."
echo "发送流式请求..."
curl -s -X POST "http://$HOST:$PORT/v1/chat/completions" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen2.5-7b",
        "messages": [
            {"role": "user", "content": "数到5"}
        ],
        "max_tokens": 50,
        "stream": true
    }' | head -10

echo ""
echo "================================"
echo "测试完成！"
