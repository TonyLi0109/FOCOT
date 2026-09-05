import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


JUDGE_SYSTEM_PROMPT = """You are a strict, neutral evaluator for multimodal visual reasoning experiments.
Score the candidate model output for the given MME-RealWorld multiple-choice item.

Use the image, question, answer choices, ground-truth answer, extracted prediction, and raw candidate output.
Do not reward verbosity by itself. Do not favor any method name. Score only the output quality.

Return JSON only with integer scores from 0 to 10:
- object_score: object/attribute evidence quality. Does the output identify relevant objects, attributes, counts, text, or entities?
- logic_score: reasoning quality. Does the output make a coherent inference from visual evidence to the choice?
- image_score: image grounding quality. Does the output rely on visible image evidence and spatial/visual details rather than hallucination?
- accuracy_score: final answer correctness. 10 if the final choice is exactly correct; 0 if wrong; 2 if no valid choice but partly aligned.
- total_score: rounded mean of the four scores.
- short_reason: one concise sentence explaining the score.
"""


JUDGE_SCHEMA = {
    "name": "mme_multidimensional_judgment",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "object_score": {"type": "integer", "minimum": 0, "maximum": 10},
            "logic_score": {"type": "integer", "minimum": 0, "maximum": 10},
            "image_score": {"type": "integer", "minimum": 0, "maximum": 10},
            "accuracy_score": {"type": "integer", "minimum": 0, "maximum": 10},
            "total_score": {"type": "integer", "minimum": 0, "maximum": 10},
            "short_reason": {"type": "string"},
        },
        "required": [
            "object_score",
            "logic_score",
            "image_score",
            "accuracy_score",
            "total_score",
            "short_reason",
        ],
    },
    "strict": True,
}


def encode_image_data_url(path: str) -> Optional[str]:
    image_path = Path(path)
    if not image_path.exists():
        return None
    suffix = image_path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def make_user_prompt(row: Dict[str, Any]) -> str:
    choices = row.get("answer_choices") or []
    choice_text = "\n".join(str(choice) for choice in choices)
    return f"""Evaluate this candidate model output.

Item id: {row.get("item_id", "")}
Task: {row.get("task", "")}
Category: {row.get("category", "")}
Subtask: {row.get("subtask", "")}

Question:
{row.get("question", "")}

Answer choices:
{choice_text}

Ground-truth answer: {row.get("answer", "")}
Extracted candidate prediction: {row.get("prediction", "")}

Candidate raw output:
{row.get("raw_output", "")}
"""


def parse_json_response(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
        raise


def call_judge(
    client: OpenAI,
    model: str,
    row: Dict[str, Any],
    include_image: bool,
    max_retries: int,
    sleep_seconds: float,
) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": make_user_prompt(row)}]
    if include_image:
        data_url = encode_image_data_url(row.get("image", ""))
        if data_url:
            content.insert(0, {"type": "image_url", "image_url": {"url": data_url}})

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": JUDGE_SCHEMA,
                },
                temperature=0,
            )
            message = response.choices[0].message.content or "{}"
            return parse_json_response(message)
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= max_retries:
                break
            time.sleep(sleep_seconds * (2**attempt))
    raise RuntimeError(f"judge call failed after {max_retries} attempts: {last_error}")


def load_predictions(path: str, limit: int, category: str = "") -> List[Dict[str, Any]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if category:
        keep = {item.strip() for item in category.split(",") if item.strip()}
        rows = [row for row in rows if row.get("category") in keep]
    if limit > 0:
        rows = rows[:limit]
    return rows


def summarize(judged_rows: List[Dict[str, Any]], scale: float) -> Dict[str, Any]:
    if not judged_rows:
        return {
            "num_records": 0,
            "raw_0_10": {"Obj.": 0, "Log.": 0, "Ima.": 0, "Acc.": 0, "Total": 0},
            "paper_row_0_1_total_4": {"Obj.": 0, "Log.": 0, "Ima.": 0, "Acc.": 0, "Total": 0},
            "scaled_row": {"Obj.": 0, "Log.": 0, "Ima.": 0, "Acc.": 0, "Total": 0},
        }

    def avg(field: str) -> float:
        return sum(float(row["judge"][field]) for row in judged_rows) / len(judged_rows)

    raw = {
        "Obj.": round(avg("object_score"), 3),
        "Log.": round(avg("logic_score"), 3),
        "Ima.": round(avg("image_score"), 3),
        "Acc.": round(avg("accuracy_score"), 3),
        "Total": round(avg("total_score"), 3),
    }
    paper = {
        "Obj.": round(raw["Obj."] / 10.0, 2),
        "Log.": round(raw["Log."] / 10.0, 2),
        "Ima.": round(raw["Ima."] / 10.0, 2),
        "Acc.": round(raw["Acc."] / 10.0, 2),
    }
    paper["Total"] = round(paper["Obj."] + paper["Log."] + paper["Ima."] + paper["Acc."], 2)
    scaled = {key: round(value * scale, 2) for key, value in raw.items()}
    return {
        "num_records": len(judged_rows),
        "raw_0_10": raw,
        "paper_row_0_1_total_4": paper,
        "scaled_row": scaled,
    }


def resolve_api_key(args: argparse.Namespace) -> str:
    value = args.api_key_env.strip()
    api_key = os.environ.get(value, "")
    if not api_key:
        raise RuntimeError(
            f"OpenAI API key not found. Set environment variable {value}, "
            "before running the judge."
        )
    return api_key


def evaluate_file(args: argparse.Namespace, method_name: str, pred_path: str) -> Dict[str, Any]:
    client = OpenAI(api_key=resolve_api_key(args))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    judged_path = output_dir / f"{method_name}_judge_rows.json"

    rows = load_predictions(pred_path, args.limit, args.category)
    judged_rows: List[Dict[str, Any]] = []
    done_ids = set()
    if args.resume and judged_path.exists():
        judged_rows = json.loads(judged_path.read_text(encoding="utf-8"))
        done_ids = {row["item_id"] for row in judged_rows}
        rows = [row for row in rows if row.get("item_id") not in done_ids]
        print(f"[{method_name}] resume: loaded {len(done_ids)} judged rows, {len(rows)} remaining")

    for index, row in enumerate(rows, 1):
        judge = call_judge(
            client=client,
            model=args.judge_model,
            row=row,
            include_image=not args.no_image,
            max_retries=args.max_retries,
            sleep_seconds=args.sleep_seconds,
        )
        judged_rows.append(
            {
                "item_id": row.get("item_id"),
                "method": method_name,
                "category": row.get("category"),
                "task": row.get("task"),
                "answer": row.get("answer"),
                "prediction": row.get("prediction"),
                "correct": row.get("correct"),
                "judge": judge,
            }
        )
        if index % args.save_every == 0:
            judged_path.write_text(json.dumps(judged_rows, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[{method_name}] saved {len(judged_rows)} total ({index}/{len(rows)} new)")

    judged_path.write_text(json.dumps(judged_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = summarize(judged_rows, args.scale)
    summary.update(
        {
            "method": method_name,
            "prediction_file": pred_path,
            "judge_model": args.judge_model,
            "include_image": not args.no_image,
        }
    )
    (output_dir / f"{method_name}_judge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def parse_method_arg(value: str) -> Dict[str, str]:
    methods = {}
    for chunk in value:
        if "=" not in chunk:
            raise ValueError("--method_file must be NAME=PATH")
        name, path = chunk.split("=", 1)
        methods[name.strip()] = path.strip()
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(description="Use an OpenAI judge model to compute Table 1 multidimensional MME scores.")
    parser.add_argument(
        "--method_file",
        nargs="+",
        required=True,
        help="One or more NAME=prediction_json pairs, e.g. FOCOT=outputs/.../focot_predictions.json",
    )
    parser.add_argument("--output_dir", default="outputs/api_judge_mme_table")
    parser.add_argument("--judge_model", default="gpt-5.4-mini")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--category", default="", help="Optional comma-separated category filter.")
    parser.add_argument("--scale", type=float, default=10.0, help="Optional: multiply raw 0-10 scores into scaled_row.")
    parser.add_argument("--no_image", action="store_true", help="Judge from text/labels only. Cheaper, but less visual.")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--sleep_seconds", type=float, default=2.0)
    args = parser.parse_args()

    resolve_api_key(args)

    methods = parse_method_arg(args.method_file)
    summaries = {}
    for method_name, pred_path in methods.items():
        summaries[method_name] = evaluate_file(args, method_name, pred_path)

    table = {
        method_name: summary["paper_row_0_1_total_4"]
        for method_name, summary in summaries.items()
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "table1_rows.json").write_text(
        json.dumps(table, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(table, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
