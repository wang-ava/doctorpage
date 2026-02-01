#!/usr/bin/env bash
# CheXpert/NegBio 标注器安装脚本（安装到脚本同级目录）
# 用法: bash setup_labelers.sh [chexpert]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$SCRIPT_DIR"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-chexpert-label}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PIP_BIN="${PIP_BIN:-pip}"

cd "$INSTALL_DIR"

ensure_conda() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "未找到 conda：请先安装/初始化 conda（例如先执行 'conda init bash' 并重新打开终端），或在已激活 conda 的 shell 中运行本脚本。"
        exit 1
    fi

    local conda_base
    conda_base="$(conda info --base)"

    # 让 'conda activate' 在非交互 shell 中可用
    # shellcheck disable=SC1090
    source "${conda_base}/etc/profile.d/conda.sh"
}

install_chexpert() {
    echo "=========================================="
    echo "安装 CheXpert 标注器"
    echo "=========================================="

    # 安装 NegBio（依赖）
    if [ -d "NegBio" ]; then
        echo "NegBio 目录已存在，跳过克隆"
    else
        git clone https://github.com/ncbi-nlp/NegBio.git
    fi

    # 设置环境变量
    echo ""
    echo "请添加以下环境变量到你的 ~/.bashrc 或 ~/.zshrc："
    echo "export PYTHONPATH=${INSTALL_DIR}/NegBio:\$PYTHONPATH"

    # 安装 CheXpert 标注器
    if [ -d "chexpert-labeler" ]; then
        echo "chexpert-labeler 目录已存在，跳过克隆"
    else
        git clone https://github.com/stanfordmlgroup/chexpert-labeler.git
    fi

    cd chexpert-labeler

    ensure_conda

    if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
        echo "conda 环境已存在：${CONDA_ENV_NAME}（跳过创建）"
    else
        conda env create -n "${CONDA_ENV_NAME}" -f environment.yml
    fi

    conda activate "${CONDA_ENV_NAME}"

    python -m nltk.downloader universal_tagset punkt wordnet
    python -c "from bllipparser import RerankingParser; RerankingParser.fetch_and_load('GENIA+PubMed')"

    conda deactivate

    cd ..
    echo "CheXpert 标注器安装完成！"
}

case "${1:-chexpert}" in
    chexpert)
        install_chexpert
        ;;
    *)
        echo "用法: bash setup_chexpert_labeler.sh [chexpert]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo "安装目录: ${INSTALL_DIR}"
echo ""
echo "使用方法:"
echo "  1) 激活环境: conda activate ${CONDA_ENV_NAME}"
echo "  2) 运行标注: python ${INSTALL_DIR}/chexpert-labeler/label.py --reports_path <reports.csv> --output_path <output.csv>"
