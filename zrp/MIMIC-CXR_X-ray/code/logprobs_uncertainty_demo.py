"""
Logprobs 不确定性分析 Demo
============================
演示如何使用 logprobs 来分析 LLM 输出的不确定性

支持:
- OpenAI API
- OpenRouter API (支持多种模型)
"""

import os
import math
import json
from typing import Optional
from openai import OpenAI


def get_client(provider: str = "openrouter") -> OpenAI:
    """
    获取API客户端

    Args:
        provider: "openai" 或 "openrouter"
    """
    if provider == "openrouter":
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    else:
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def query_with_logprobs(
    client: OpenAI,
    prompt: str,
    model: str = "gpt-4o-mini",
    top_logprobs: int = 5,
    max_tokens: int = 100,
) -> dict:
    """
    带 logprobs 的查询

    Args:
        client: OpenAI客户端
        prompt: 用户问题
        model: 模型名称
        top_logprobs: 返回概率最高的前k个候选token
        max_tokens: 最大生成token数

    Returns:
        包含响应和logprobs信息的字典
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=top_logprobs,
    )

    return {
        "content": response.choices[0].message.content,
        "logprobs": response.choices[0].logprobs,
        "model": response.model,
    }


def logprob_to_prob(logprob: float) -> float:
    """将logprob转换为概率"""
    return math.exp(logprob)


def analyze_uncertainty(logprobs_data) -> dict:
    """
    分析输出的不确定性

    Args:
        logprobs_data: API返回的logprobs对象

    Returns:
        不确定性分析结果
    """
    if logprobs_data is None or logprobs_data.content is None:
        return {"error": "No logprobs data available"}

    tokens_info = []
    total_logprob = 0
    low_confidence_tokens = []

    for token_data in logprobs_data.content:
        token = token_data.token
        logprob = token_data.logprob
        prob = logprob_to_prob(logprob)
        total_logprob += logprob

        # 收集top候选tokens
        top_alternatives = []
        if token_data.top_logprobs:
            for alt in token_data.top_logprobs:
                top_alternatives.append({
                    "token": alt.token,
                    "logprob": alt.logprob,
                    "prob": logprob_to_prob(alt.logprob),
                })

        token_info = {
            "token": token,
            "logprob": logprob,
            "prob": prob,
            "prob_percent": f"{prob * 100:.2f}%",
            "top_alternatives": top_alternatives,
        }
        tokens_info.append(token_info)

        # 标记低置信度token (概率 < 50%)
        if prob < 0.5:
            low_confidence_tokens.append(token_info)

    # 计算整体指标
    n_tokens = len(tokens_info)
    avg_logprob = total_logprob / n_tokens if n_tokens > 0 else 0
    avg_prob = logprob_to_prob(avg_logprob)

    # 计算perplexity (困惑度) - 越低表示模型越确定
    perplexity = math.exp(-total_logprob / n_tokens) if n_tokens > 0 else float('inf')

    return {
        "total_tokens": n_tokens,
        "total_logprob": total_logprob,
        "avg_logprob": avg_logprob,
        "avg_prob": avg_prob,
        "avg_prob_percent": f"{avg_prob * 100:.2f}%",
        "perplexity": perplexity,
        "low_confidence_count": len(low_confidence_tokens),
        "low_confidence_ratio": len(low_confidence_tokens) / n_tokens if n_tokens > 0 else 0,
        "tokens_detail": tokens_info,
        "low_confidence_tokens": low_confidence_tokens,
    }


def print_analysis(result: dict, analysis: dict, show_all_tokens: bool = False):
    """
    打印分析结果
    """
    print("=" * 60)
    print(f"模型: {result['model']}")
    print("=" * 60)
    print(f"\n回答: {result['content']}\n")
    print("-" * 60)
    print("不确定性分析:")
    print("-" * 60)
    print(f"  总token数: {analysis['total_tokens']}")
    print(f"  平均概率: {analysis['avg_prob_percent']}")
    print(f"  困惑度 (Perplexity): {analysis['perplexity']:.4f}")
    print(f"  低置信度token数: {analysis['low_confidence_count']} ({analysis['low_confidence_ratio']*100:.1f}%)")

    if analysis['low_confidence_tokens']:
        print("\n低置信度tokens (概率 < 50%):")
        for t in analysis['low_confidence_tokens'][:10]:  # 只显示前10个
            print(f"  '{t['token']}': {t['prob_percent']}")
            if t['top_alternatives']:
                alt_str = ", ".join([f"'{a['token']}'({a['prob']*100:.1f}%)"
                                     for a in t['top_alternatives'][:3]])
                print(f"    候选: {alt_str}")

    if show_all_tokens:
        print("\n所有tokens详情:")
        for t in analysis['tokens_detail']:
            print(f"  '{t['token']}': {t['prob_percent']}")


def uncertainty_score(analysis: dict) -> str:
    """
    根据分析结果给出不确定性评级
    """
    perplexity = analysis['perplexity']
    low_conf_ratio = analysis['low_confidence_ratio']

    if perplexity < 1.5 and low_conf_ratio < 0.1:
        return "高置信度 - 模型非常确定"
    elif perplexity < 3 and low_conf_ratio < 0.3:
        return "中等置信度 - 模型较为确定"
    elif perplexity < 10:
        return "低置信度 - 模型不太确定，建议核实"
    else:
        return "极低置信度 - 可能存在幻觉，需要人工审核"


# ============================================================
# 示例用法
# ============================================================

def demo_factual_vs_uncertain():
    """
    演示: 事实性问题 vs 模糊问题的不确定性差异
    """
    print("\n" + "=" * 60)
    print("Demo: 事实性问题 vs 模糊问题的不确定性对比")
    print("=" * 60)

    # 使用OpenAI (需要设置 OPENAI_API_KEY)
    # 或使用OpenRouter (需要设置 OPENROUTER_API_KEY)

    try:
        client = get_client("openai")
        model = "gpt-4o-mini"
    except Exception:
        print("请设置 OPENAI_API_KEY 或 OPENROUTER_API_KEY 环境变量")
        return

    # 测试问题
    questions = [
        "What is 2 + 2?",  # 简单事实，应该高置信度
        "What is the capital of France?",  # 明确事实
        "Who will win the next election?",  # 不确定问题
        "What is the best programming language?",  # 主观问题
    ]

    for q in questions:
        print(f"\n问题: {q}")
        try:
            result = query_with_logprobs(client, q, model=model)
            analysis = analyze_uncertainty(result['logprobs'])

            print(f"回答: {result['content'][:100]}...")
            print(f"困惑度: {analysis['perplexity']:.4f}")
            print(f"低置信度比例: {analysis['low_confidence_ratio']*100:.1f}%")
            print(f"评级: {uncertainty_score(analysis)}")
        except Exception as e:
            print(f"Error: {e}")


def demo_hallucination_detection():
    """
    演示: 利用logprobs检测潜在的幻觉
    """
    print("\n" + "=" * 60)
    print("Demo: 幻觉检测")
    print("=" * 60)

    try:
        client = get_client("openai")
        model = "gpt-4o-mini"
    except Exception:
        print("请设置 OPENAI_API_KEY 环境变量")
        return

    # 询问一个可能导致幻觉的问题
    prompt = "Tell me about the research paper 'Deep Learning for Medical Image Analysis' published in Nature in 2019 by Dr. John Smith."

    print(f"问题: {prompt}")

    try:
        result = query_with_logprobs(client, prompt, model=model, max_tokens=200)
        analysis = analyze_uncertainty(result['logprobs'])

        print_analysis(result, analysis)
        print(f"\n不确定性评级: {uncertainty_score(analysis)}")

        # 幻觉警告
        if analysis['perplexity'] > 5 or analysis['low_confidence_ratio'] > 0.4:
            print("\n⚠️  警告: 该回答可能包含幻觉内容，建议核实!")
    except Exception as e:
        print(f"Error: {e}")


def demo_medical_qa():
    """
    演示: 医学问答中的不确定性分析
    """
    print("\n" + "=" * 60)
    print("Demo: 医学问答不确定性分析")
    print("=" * 60)

    try:
        client = get_client("openai")
        model = "gpt-4o-mini"
    except Exception:
        print("请设置 OPENAI_API_KEY 环境变量")
        return

    # 医学问题示例
    questions = [
        "What are common symptoms of pneumonia?",  # 常见知识
        "What is the treatment for stage 4 pancreatic cancer?",  # 复杂问题
    ]

    for q in questions:
        print(f"\n问题: {q}")
        try:
            result = query_with_logprobs(client, q, model=model, max_tokens=150)
            analysis = analyze_uncertainty(result['logprobs'])

            print(f"回答: {result['content'][:200]}...")
            print(f"困惑度: {analysis['perplexity']:.4f}")
            print(f"不确定性评级: {uncertainty_score(analysis)}")
        except Exception as e:
            print(f"Error: {e}")


def get_default_provider_and_model():
    """自动检测可用的provider和对应模型"""
    if os.getenv("OPENROUTER_API_KEY"):
        # OpenRouter 支持 logprobs 的模型
        return "openrouter", "openai/gpt-4o-mini"
    elif os.getenv("OPENAI_API_KEY"):
        return "openai", "gpt-4o-mini"
    else:
        return None, None


if __name__ == "__main__":
    print("Logprobs 不确定性分析 Demo")
    print("=" * 60)
    print("请确保设置了以下环境变量之一:")
    print("  - OPENROUTER_API_KEY (使用OpenRouter)")
    print("  - OPENAI_API_KEY (使用OpenAI)")
    print("=" * 60)

    # 简单交互式示例
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What is the capital of France?"

    print(f"\n测试问题: {question}")

    provider, model = get_default_provider_and_model()
    if provider is None:
        print("Error: 请设置 OPENROUTER_API_KEY 或 OPENAI_API_KEY 环境变量")
        sys.exit(1)

    print(f"使用: {provider} / {model}")

    try:
        client = get_client(provider)
        result = query_with_logprobs(client, question, model=model)
        # print("result: ", result)
        analysis = analyze_uncertainty(result['logprobs'])
        print_analysis(result, analysis)
        print(f"\n不确定性评级: {uncertainty_score(analysis)}")
    except Exception as e:
        print(f"Error: {e}")
