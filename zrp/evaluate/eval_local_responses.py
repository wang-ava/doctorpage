"""
评估本地模型生成的回答

使用方法:
1. 设置环境变量:
   export OPENROUTER_API_KEY="your-openrouter-api-key"

2. 运行评估:
   cd /net/scratch/zhaorun/zichen/zrp/nus_med/yqw/Healthbench
   python evaluate/eval_local_responses.py \
       --input dataset/hard_2025-05-08-21-00-10_english_only_sample_100.jsonl \
       --responses result/response/OpenGVLab_InternVL2-8B/EN/round1.json \
       --output result/eval/OpenGVLab_InternVL2-8B/EN/round1_eval.json

3. 可选参数:
   --debug        调试模式（只运行5个样本）
   --examples N   指定运行的样本数量
   --n-threads N  并行线程数
   --grader-model 指定评分模型（默认 openai/gpt-4o-mini）
"""

import argparse
import json
import os
import sys
import time
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
from openai import OpenAI
import openai
from tqdm import tqdm
import jinja2


# ============== 类型定义 ==============
Message = dict[str, Any]
MessageList = list[Message]


@dataclass
class SamplerResponse:
    response_text: str
    actual_queried_message_list: MessageList
    response_metadata: dict[str, Any]


@dataclass
class SingleEvalResult:
    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    html: str | None = None
    convo: MessageList | None = None
    example_level_metadata: dict[str, Any] | None = None


@dataclass
class EvalResult:
    score: float | None
    metrics: dict[str, float] | None
    htmls: list[str]
    convos: list[MessageList]
    metadata: dict[str, Any] | None


# ============== OpenRouter Sampler ==============
class OpenRouterSampler:
    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        api_key: str | None = None,
    ):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                {"role": "system", "content": self.system_message}
            ] + message_list

        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenRouter API returned empty response")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            except openai.BadRequestError as e:
                print(f"Bad Request Error: {e}")
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial
                print(f"Error, retry {trial} after {exception_backoff}s: {e}")
                time.sleep(exception_backoff)
                trial += 1
                if trial > 5:
                    return SamplerResponse(
                        response_text="No response (max retries).",
                        response_metadata={"usage": None},
                        actual_queried_message_list=message_list,
                    )


# ============== 评分模板 ==============
GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


# ============== RubricItem 类 ==============
class RubricItem:
    """
    [
        {
            "criterion": "Recognizes that in the healthcare context of ulcerative colitis, the presenting symptoms are a flare-up due to increased frequency, blood, and pain, and recommends an urgent healthcare review.",
            "points": 9,
            "tags": ["level:example", "axis:completeness"]
        },
        {
            "criterion": "Considers other probable causes through gastroenterology testing of stool, complete blood count, inflammatory markers, and renal function tests.",
            "points": 7,
            "tags": ["level:example", "axis:completeness"]
        }
    ]
    """
    def __init__(self, criterion: str, points: float, tags: list[str]):
        self.criterion = criterion
        self.points = points
        self.tags = tags

    def __str__(self):
        return f"[{self.points}] {self.criterion}"

    def to_dict(self):
        return {
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            criterion=d["criterion"],
            points=d["points"],
            tags=d["tags"],
        )


# ============== 工具函数 ==============
def parse_json_to_dict(json_string: str) -> dict:
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())
    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        return {}


def calculate_score(
    rubric_items: list[RubricItem], grading_response_list: list[dict]
) -> float | None:
    total_possible_points = sum(
        rubric_item.points for rubric_item in rubric_items if rubric_item.points > 0
    )
    if total_possible_points == 0:
        return None

    achieved_points = sum(
        rubric_item.points
        for rubric_item, grading_response in zip(rubric_items, grading_response_list, strict=True)
        if grading_response.get("criteria_met", False)
    )
    return achieved_points / total_possible_points


def map_with_progress(fn, items, num_threads=10, pbar=True):
    """多线程执行并显示进度"""
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(fn, item): i for i, item in enumerate(items)}
        iter_futures = tqdm(as_completed(futures), total=len(items), disable=not pbar)
        for future in iter_futures:
            idx = futures[future]
            results[idx] = future.result()
    return results


def _compute_clipped_stats(values: list, stat: str):
    if stat == "mean":
        return float(np.clip(np.mean(values), 0, 1))
    elif stat == "n_samples":
        return len(values)
    elif stat == "bootstrap_std":
        bootstrap_samples = [np.random.choice(values, len(values)) for _ in range(1000)]
        bootstrap_means = [_compute_clipped_stats(list(s), "mean") for s in bootstrap_samples]
        return float(np.std(bootstrap_means))
    else:
        raise ValueError(f"Unknown stat: {stat}")


def aggregate_results(single_eval_results: list[SingleEvalResult]) -> EvalResult:
    """汇总评估结果"""
    name2values = defaultdict(list)
    htmls = []
    convos = []
    metadata = []

    for result in single_eval_results:
        for name, value in result.metrics.items():
            name2values[name].append(value)
        if result.score is not None:
            name2values["score"].append(result.score)
        htmls.append(result.html)
        convos.append(result.convo)
        metadata.append(result.example_level_metadata)

    final_metrics = {}
    for name, values in name2values.items():
        for stat in ["mean", "n_samples", "bootstrap_std"]:
            key = name if stat == "mean" else f"{name}:{stat}"
            final_metrics[key] = _compute_clipped_stats(values, stat)

    return EvalResult(
        score=final_metrics.pop("score", None),
        metrics=final_metrics,
        htmls=htmls,
        convos=convos,
        metadata={"example_level_metadata": metadata},
    )


# ============== HTML 模板 ==============
HTML_JINJA = """
<h3>Prompt messages</h3>
{% for message in prompt_messages %}
<p><strong>{{ message.role }}:</strong> {{ message.content | e }}</p>
{% endfor %}
<h3>Response</h3>
<p><strong>{{ next_message.role }}:</strong> {{ next_message.content | e }}</p>
<h3>Score: {{ score }}</h3>
<p>Rubrics with grades: {{ rubric_grades }}</p>
""".strip()


def make_report(eval_result: EvalResult) -> str:
    """生成 HTML 报告"""
    score_str = f"{eval_result.score:.4f}" if eval_result.score is not None else "N/A"
    html = f"""
    <html>
    <head><title>HealthBench Evaluation Report</title></head>
    <body>
    <h1>HealthBench Evaluation Report</h1>
    <h2>Overall Score: {score_str}</h2>
    <h2>Metrics</h2>
    <ul>
    """
    if eval_result.metrics:
        for k, v in sorted(eval_result.metrics.items()):
            v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            html += f"<li>{k}: {v_str}</li>"
    html += "</ul><hr>"

    for i, h in enumerate(eval_result.htmls or []):
        html += f"<h2>Example {i+1}</h2>{h}<hr>"

    html += "</body></html>"
    return html


# ============== 主函数 ==============
def load_input_data(input_path: str) -> dict:
    """加载输入数据"""
    data_map = {}
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            prompt_id = item['prompt_id']
            item['rubrics'] = [RubricItem.from_dict(r) for r in item['rubrics']]
            data_map[prompt_id] = item
    return data_map


def load_responses(response_path: str) -> dict:
    """加载模型回答"""
    response_map = {}
    with open(response_path, 'r', encoding='utf-8') as f:
        responses = json.load(f)
    for item in responses:
        prompt_id = item['prompt_id']
        response_map[prompt_id] = item
    return response_map


def grade_sample(
    grader_model,
    prompt: list[dict[str, str]],
    response_text: str,
    example_tags: list[str],
    rubric_items: list[RubricItem],
) -> tuple[dict, str, list[dict]]:
    """评分单个样本"""
    convo_with_response = prompt + [dict(content=response_text, role="assistant")]

    def grade_rubric_item(rubric_item: RubricItem) -> dict:
        convo_str = "\n\n".join(
            [f"{m['role']}: {m['content']}" for m in convo_with_response]
        )
        grader_prompt = GRADER_TEMPLATE.replace(
            "<<conversation>>", convo_str
        ).replace("<<rubric_item>>", str(rubric_item))
        messages = [dict(content=grader_prompt, role="user")]

        for retry in range(5):
            try:
                sampler_response = grader_model(messages)
                grading_response_dict = parse_json_to_dict(sampler_response.response_text)
                if "criteria_met" in grading_response_dict:
                    label = grading_response_dict["criteria_met"]
                    if label is True or label is False:
                        return grading_response_dict
                print(f"Bad JSON, retrying ({retry+1}/5)...")
            except Exception as e:
                print(f"Error: {e}, retrying ({retry+1}/5)...")

        return {"criteria_met": False, "explanation": "Grading failed"}

    # 顺序评分每个 rubric (避免过多并发)
    grading_response_list = [grade_rubric_item(r) for r in rubric_items]

    # 计算总分
    overall_score = calculate_score(rubric_items, grading_response_list)
    if overall_score is None:
        overall_score = 0.0

    metrics = {"overall_score": overall_score}

    # example-level tags
    for tag in example_tags:
        metrics[tag] = overall_score

    # rubric-level tags
    rubric_tag_items_grades = defaultdict(list)
    for rubric_item, grading_response in zip(rubric_items, grading_response_list):
        for tag in rubric_item.tags:
            rubric_tag_items_grades[tag].append((rubric_item, grading_response))

    for tag, items_grades in rubric_tag_items_grades.items():
        items, grades = zip(*items_grades)
        score = calculate_score(list(items), list(grades))
        if score is not None:
            metrics[tag] = score

    # 构建解释
    rubric_items_with_grades = []
    readable_list = []
    for rubric_item, grading_response in zip(rubric_items, grading_response_list):
        explanation = grading_response.get("explanation", "No explanation")
        criteria_met = grading_response.get("criteria_met", False)
        readable_list.append(f"[{criteria_met}] {rubric_item}\n  Explanation: {explanation}")
        rubric_items_with_grades.append({
            **rubric_item.to_dict(),
            "criteria_met": criteria_met,
            "explanation": explanation,
        })

    readable_str = "\n\n".join(readable_list)
    return metrics, readable_str, rubric_items_with_grades


def main():
    parser = argparse.ArgumentParser(description="评估本地模型生成的回答")
    parser.add_argument("--input", type=str, required=True, help="输入数据文件路径 (JSONL)")
    parser.add_argument("--responses", type=str, required=True, help="模型回答文件路径 (JSON)")
    parser.add_argument("--output", type=str, help="输出结果文件路径")
    parser.add_argument("--grader-model", type=str, default="openai/gpt-5.2", help="评分模型")
    parser.add_argument("--n-threads", type=int, default=10, help="并行线程数")
    parser.add_argument("--debug", action="store_true", help="调试模式 (5个样本)")
    parser.add_argument("--examples", type=int, help="样本数量")

    args = parser.parse_args()

    # 检查 API Key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("错误: 请设置 OPENROUTER_API_KEY 环境变量")
        print("  export OPENROUTER_API_KEY='your-api-key'")
        return

    print(f"加载输入数据: {args.input}")
    input_data = load_input_data(args.input)
    print(f"  共 {len(input_data)} 条")

    print(f"加载模型回答: {args.responses}")
    responses = load_responses(args.responses)
    print(f"  共 {len(responses)} 条")

    # 匹配数据
    matched = []
    for prompt_id, inp in input_data.items():
        if prompt_id in responses:
            matched.append({
                "prompt_id": prompt_id,
                "prompt": inp["prompt"],
                "rubrics": inp["rubrics"],
                "example_tags": inp.get("example_tags", []),
                "response": responses[prompt_id]["response"],
            })

    print(f"匹配: {len(matched)} 条")
    if not matched:
        print("错误: 没有匹配的数据")
        return

    # 限制数量
    if args.debug:
        matched = matched[:5]
        print(f"调试模式: {len(matched)} 条")
    elif args.examples:
        matched = matched[:args.examples]
        print(f"限制: {len(matched)} 条")

    # 初始化评分模型
    print(f"评分模型: {args.grader_model}")
    grader = OpenRouterSampler(
        model=args.grader_model,
        system_message="You are a helpful assistant.",
        max_tokens=2048,
    )

    # 评估函数
    def evaluate_one(example):
        metrics, explanation, rubric_grades = grade_sample(
            grader_model=grader,
            prompt=example["prompt"],
            response_text=example["response"],
            example_tags=example["example_tags"],
            rubric_items=example["rubrics"],
        )
        score = metrics["overall_score"]

        html = jinja2.Template(HTML_JINJA).render(
            prompt_messages=example["prompt"],
            next_message={"role": "assistant", "content": example["response"]},
            score=score,
            rubric_grades=explanation.replace("\n", "<br>"),
        )

        return SingleEvalResult(
            html=html,
            score=score,
            convo=example["prompt"] + [{"role": "assistant", "content": example["response"]}],
            metrics=metrics,
            example_level_metadata={
                "score": score,
                "rubric_items": rubric_grades,
                "prompt_id": example["prompt_id"],
            },
        )

    # 运行评估
    print(f"\n开始评估 (线程数: {args.n_threads})...")
    results = map_with_progress(evaluate_one, matched, num_threads=args.n_threads, pbar=True)

    # 汇总
    final = aggregate_results(results)

    # 输出路径
    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    response_path = Path(args.responses)
    model_name = response_path.parent.parent.name.replace("/", "_")
    round_name = response_path.stem

    if args.output:
        output_path = Path(args.output)
    else:
        # 保存到 evaluate 目录下
        script_dir = Path(__file__).parent
        output_path = script_dir / f"output/healthbench_{model_name}_{round_name}_{date_str}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存
    metrics = dict(sorted((final.metrics or {}).items()))
    metrics["score"] = final.score

    print(f"\n{'='*60}")
    print(f"Overall Score: {final.score:.4f}")
    print(f"{'='*60}")

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"指标: {output_path}")

    full_path = output_path.with_name(output_path.stem + "_full.json")
    with open(full_path, 'w') as f:
        json.dump({
            "score": final.score,
            "metrics": final.metrics,
            "metadata": final.metadata,
        }, f, indent=2, ensure_ascii=False)
    print(f"完整: {full_path}")

    html_path = output_path.with_suffix(".html")
    with open(html_path, 'w') as f:
        f.write(make_report(final))
    print(f"报告: {html_path}")


if __name__ == "__main__":
    main()
