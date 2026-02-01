#!/bin/bash
# 批量评估所有轮次的脚本
#
# 使用方法:
#   1. 设置环境变量:
#      export OPENROUTER_API_KEY="your-openrouter-api-key"
#
#   2. 运行脚本:
#      cd /net/scratch/zhaorun/zichen/zrp/nus_med/yqw/Healthbench
#      bash evaluate/eval_all_rounds.sh
#
#   3. 调试模式 (每轮只运行5个样本):
#      bash evaluate/eval_all_rounds.sh --debug

set -e

# 配置组合列表：每一项是 INPUT_FILE|RESPONSE_DIR|OUTPUT_DIR
COMBOS=(
#   "evaluate/input/dataset_format_english.jsonl|result/response/openai_gpt-5.2/EN|result/eval/openai_gpt-5.2/EN"
#   "evaluate/input/dataset_format_english.jsonl|result/response/openai_gpt-5.2/ZH|result/eval/openai_gpt-5.2/ZH_english"
#   "evaluate/input/dataset_format_chinese.jsonl|result/response/openai_gpt-5.2/ZH|result/eval/openai_gpt-5.2/ZH_chinese"
#   "evaluate/input/dataset_format_english.jsonl|result/response/openai_gpt-5.2/MS|result/eval/openai_gpt-5.2/MS_english"
#   "evaluate/input/dataset_format_malay.jsonl|result/response/openai_gpt-5.2/MS|result/eval/openai_gpt-5.2/MS_malay"
#   "evaluate/input/dataset_format_english.jsonl|result/response/openai_gpt-5.2/TH|result/eval/openai_gpt-5.2/TH_english"
  "evaluate/input/dataset_format_thai.jsonl|result/response/openai_gpt-5.2/TH|result/eval/openai_gpt-5.2/TH_thai"
)

# 解析参数
DEBUG_FLAG=""
if [[ "$1" == "--debug" ]]; then
    DEBUG_FLAG="--debug"
    echo "调试模式: 每轮只运行5个样本"
fi

# 检查 API Key
if [[ -z "${OPENROUTER_API_KEY}" ]]; then
    echo "错误: 请设置 OPENROUTER_API_KEY 环境变量"
    echo "  export OPENROUTER_API_KEY='your-api-key'"
    exit 1
fi

# 遍历所有组合
for combo in "${COMBOS[@]}"; do
    IFS='|' read -r INPUT_FILE RESPONSE_DIR OUTPUT_DIR <<< "${combo}"

    echo ""
    echo "=========================================="
    echo "开始评估组合:"
    echo "  INPUT_FILE  = ${INPUT_FILE}"
    echo "  RESPONSE_DIR= ${RESPONSE_DIR}"
    echo "  OUTPUT_DIR  = ${OUTPUT_DIR}"
    echo "=========================================="

    # 创建输出目录
    mkdir -p "${OUTPUT_DIR}"
    # 评估每一轮
    for round in 2 3; do
        echo ""
        echo "=========================================="
        echo "评估 Round ${round}"
        echo "=========================================="

        RESPONSE_FILE="${RESPONSE_DIR}/round${round}.json"
        OUTPUT_FILE="${OUTPUT_DIR}/round${round}_eval.json"

        if [[ ! -f "${RESPONSE_FILE}" ]]; then
            echo "警告: 回答文件不存在: ${RESPONSE_FILE}"
            continue
        fi

        python evaluate/eval_local_responses.py \
            --input "${INPUT_FILE}" \
            --responses "${RESPONSE_FILE}" \
            --output "${OUTPUT_FILE}" \
            --n-threads 10 \
            ${DEBUG_FLAG}
    done

    # 显示该组合的汇总结果
    echo ""
    echo "该组合各轮次分数汇总:"
    for round in 1 2 3; do
        OUTPUT_FILE="${OUTPUT_DIR}/round${round}_eval.json"
        if [[ -f "${OUTPUT_FILE}" ]]; then
            SCORE=$(python3 -c "import json; print(json.load(open('${OUTPUT_FILE}'))['score'])" 2>/dev/null || echo "N/A")
            echo "  Round ${round}: ${SCORE}"
        else
            echo "  Round ${round}: N/A (no output file)"
        fi
    done
done

echo ""
echo "=========================================="
echo "所有组合评估完成!"
echo "=========================================="