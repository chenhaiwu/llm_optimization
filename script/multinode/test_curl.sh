#!/bin/bash

curl --location 'http://127.0.0.1:8000/v1/chat/completions' \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/data0/models/deepseek-ai/DeepSeek-R1",
        "stream": false,
	"temperature":0.6,
        "messages": [
            {"role": "user", "content": "中国的首都是哪？"}
        ],
        "top_p":0.95
    }'
