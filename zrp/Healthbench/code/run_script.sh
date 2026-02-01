# 创建专门用于 vLLM 推理的环境
conda create -n vllm_env python=3.10 -y
conda activate vllm_env

# 安装 vLLM（会自动安装兼容的 PyTorch 和 transformers）
pip install vllm

# 检查版本
python -c "import vllm; print(vllm.__version__)"

# 启动 vLLM 服务器，放到后台运行
nohup vllm serve OpenGVLab/InternVL2-8B \
    --trust-remote-code \
    --port 8001 \
    > vllm_server.log 2>&1 &

# 等待服务器启动（约30-60秒）
sleep 60

# 检查是否启动成功
curl http://localhost:8000/v1/models

# 或者

curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "OpenGVLab/InternVL2-8B",
      "messages": [{"role": "user", "content": "Hello"}],
      "max_tokens": 10
    }'

python code/answer-in-specific-language_v2.py \
    --mode local \
    --model "OpenGVLab/InternVL2-8B" \
    --limit-per-language 100 \
    --rounds 3