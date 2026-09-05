import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mme_qwen_experiment import (
    QwenVLRunner,
    format_choices,
    load_mme_records,
    normalize_prediction,
    score_accuracy,
)


LETTER_RE = re.compile(r"^\(([A-E])\)\s*(.*)$")


def split_choice(choice: str) -> Tuple[str, str]:
    match = LETTER_RE.match(choice.strip())
    if match:
        return match.group(1), match.group(2).strip()
    return "", choice.strip()


def infer_box(text: str) -> Optional[Tuple[float, float, float, float]]:
    lower = text.lower()
    if "doesn't feature" in lower or "does not feature" in lower or "not feature" in lower:
        return None

    x1, x2 = 0.0, 1.0
    y1, y2 = 0.0, 1.0

    if "left" in lower:
        x1, x2 = 0.0, 0.5
    if "right" in lower:
        x1, x2 = 0.5, 1.0
    if "middle left" in lower or "center left" in lower:
        x1, x2 = 0.0, 0.5
        y1, y2 = 0.25, 0.75
    if "middle right" in lower or "center right" in lower:
        x1, x2 = 0.5, 1.0
        y1, y2 = 0.25, 0.75
    if "middle" in lower or "center" in lower:
        if "left" not in lower and "right" not in lower:
            x1, x2 = 0.25, 0.75
        if "top" not in lower and "upper" not in lower and "bottom" not in lower and "lower" not in lower:
            y1, y2 = 0.25, 0.75

    if "upper" in lower or "top" in lower or "above" in lower:
        y1, y2 = 0.0, 0.5
    if "lower" in lower or "bottom" in lower:
        y1, y2 = 0.5, 1.0

    if "corner" in lower:
        if "left" not in lower and "right" not in lower:
            x1, x2 = 0.0, 0.5
        if "top" not in lower and "upper" not in lower and "bottom" not in lower and "lower" not in lower:
            y1, y2 = 0.0, 0.5

    return x1, y1, x2, y2


def region_signature(text: str) -> Tuple[str, str]:
    lower = text.lower()
    x = "center"
    y = "middle"
    if "left" in lower:
        x = "left"
    elif "right" in lower:
        x = "right"
    if "upper" in lower or "top" in lower or "above" in lower:
        y = "upper"
    elif "lower" in lower or "bottom" in lower:
        y = "lower"
    return x, y


def normalize_grounding_prediction(text: str, choices: Optional[List[str]]) -> str:
    prediction = normalize_prediction(text, choices)
    if prediction or not choices:
        return prediction
    text_sig = region_signature(text or "")
    if text_sig == ("center", "middle"):
        return ""
    matches = []
    for choice in choices:
        letter, choice_text = split_choice(choice)
        if letter and region_signature(choice_text) == text_sig:
            matches.append(letter)
    return matches[0] if len(matches) == 1 else ""


def box_text(box: Optional[Tuple[float, float, float, float]]) -> str:
    if box is None:
        return "<absent>"
    return "<box>{:.2f},{:.2f},{:.2f},{:.2f}</box>".format(*box)


def build_coordinate_prompt(record: Any) -> str:
    lines = []
    for choice in record.answer_choices or []:
        letter, text = split_choice(choice)
        if not letter:
            continue
        lines.append(f"{letter}: {box_text(infer_box(text))} {text}")
    return (
        "Answer this visual multiple-choice question using the original image plus the textual grounding "
        "coordinates below. Coordinates are normalized as x1,y1,x2,y2, where 0,0 is the upper-left corner.\n"
        "Return only one line in this exact format: Final answer: <A/B/C/D/E>.\n"
        "Do not explain.\n\n"
        f"Question: {record.question.strip()}\n\n"
        f"Answer choices:\n{format_choices(record.answer_choices)}\n\n"
        "Textual grounding candidates:\n"
        + "\n".join(lines)
    )


def draw_label(image: Any, label: str) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 38, 28), fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((11, 6), label, fill=(0, 0, 0))


def make_crop_grid(record: Any, crop_dir: Path, cell_size: int = 384) -> str:
    from PIL import Image

    source = Image.open(record.image).convert("RGB")
    width, height = source.size
    cells = []
    for choice in record.answer_choices or []:
        letter, text = split_choice(choice)
        if letter not in "ABCD":
            continue
        box = infer_box(text)
        if box is None:
            patch = Image.new("RGB", (cell_size, cell_size), (245, 245, 245))
        else:
            x1, y1, x2, y2 = box
            crop_box = (
                max(0, int(x1 * width)),
                max(0, int(y1 * height)),
                min(width, int(x2 * width)),
                min(height, int(y2 * height)),
            )
            patch = source.crop(crop_box).resize((cell_size, cell_size))
        draw_label(patch, letter)
        cells.append((letter, patch))

    while len(cells) < 4:
        patch = Image.new("RGB", (cell_size, cell_size), (245, 245, 245))
        draw_label(patch, chr(ord("A") + len(cells)))
        cells.append((chr(ord("A") + len(cells)), patch))

    grid = Image.new("RGB", (cell_size * 2, cell_size * 2), (255, 255, 255))
    for index, (_, patch) in enumerate(cells[:4]):
        grid.paste(patch, ((index % 2) * cell_size, (index // 2) * cell_size))

    crop_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", record.item_id)
    path = crop_dir / f"{safe_id}_patch_grid.jpg"
    grid.save(path, quality=92)
    return str(path)


def build_crop_prompt(record: Any) -> str:
    return (
        "The image is a 2x2 grid of cropped image patches labeled A, B, C, and D. "
        "Each patch corresponds to the similarly labeled answer option. Option E means the target is absent. "
        "Answer using only the crop evidence.\n"
        "Return only one line in this exact format: Final answer: <A/B/C/D/E>.\n"
        "Do not explain.\n\n"
        f"Question: {record.question.strip()}\n\n"
        f"Answer choices:\n{format_choices(record.answer_choices)}"
    )


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    correct = sum(1 for row in rows if row["prediction"] == row["answer"])
    latency = sum(row["latency_s"] for row in rows) / max(len(rows), 1)
    peak_memory = max((row.get("peak_gpu_memory_gb") or 0.0) for row in rows) if rows else 0.0
    return {
        "n": len(rows),
        "correct": correct,
        "score": round(correct / max(len(rows), 1) * 100, 2),
        "avg_latency_s": round(latency, 3),
        "peak_gpu_memory_gb": round(peak_memory, 3),
    }


def write_table(path: Path, summary: Dict[str, Any]) -> None:
    crop = summary["methods"]["cropped_image_patches"]
    coord = summary["methods"]["textual_coordinates"]
    text = (
        "# Grounding Mechanism Ablation\n\n"
        "Current run: MME-RealWorld-Lite `position` subset proxy, because local GQA images are not present. "
        "The comparison keeps the backbone, decoding settings, and evaluated samples fixed.\n\n"
        "| Grounding Mechanism | GQA Total Score | Avg. Latency (s/query) | Peak GPU Memory (GB) |\n"
        "|---|---:|---:|---:|\n"
        f"| Cropped Image Patches | {crop['score']:.2f}* | {crop['avg_latency_s']:.3f} | {crop['peak_gpu_memory_gb']:.3f} |\n"
        f"| Textual Coordinates (Ours) | {coord['score']:.2f}* | {coord['avg_latency_s']:.3f} | {coord['peak_gpu_memory_gb']:.3f} |\n\n"
        f"\\* Proxy score on {summary['scope']} samples, not full GQA. Replace the column label with "
        "`MME Position Score` unless GQA images are supplied and rerun.\n"
    )
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    import torch

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = output_dir / "crop_grids"
    records = [
        record
        for record in load_mme_records(args.data)
        if record.answer_choices and record.category.lower() == args.category.lower()
    ]
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise ValueError("No matching multiple-choice records found.")

    runner = QwenVLRunner(
        model_path=args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_cache=args.use_cache,
        enable_thinking=False,
    )

    methods = {
        "cropped_image_patches": [],
        "textual_coordinates": [],
    }
    progress_path = output_dir / "progress.json"

    for index, record in enumerate(records, start=1):
        for method in ("cropped_image_patches", "textual_coordinates"):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            if method == "cropped_image_patches":
                image_path = make_crop_grid(record, crop_dir)
                prompt = build_crop_prompt(record)
            else:
                image_path = record.image
                prompt = build_coordinate_prompt(record)
            raw = runner.generate(image_path, prompt)
            latency = time.perf_counter() - start
            prediction = normalize_grounding_prediction(raw, record.answer_choices)
            peak_mem = 0.0
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_reserved() / (1024 ** 3)
            row = {
                "item_id": record.item_id,
                "method": method,
                "image": record.image,
                "question": record.question,
                "answer": record.answer,
                "prediction": prediction,
                "correct": prediction == record.answer,
                "latency_s": round(latency, 4),
                "peak_gpu_memory_gb": round(peak_mem, 4),
                "raw_output": raw,
            }
            methods[method].append(row)
            (output_dir / f"{method}_predictions.json").write_text(
                json.dumps(methods[method], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            progress = {
                "completed_items": index,
                "total_items": len(records),
                "last_method": method,
                "last_item_id": record.item_id,
                "method_scores": {name: summarize(rows) for name, rows in methods.items()},
            }
            progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"[{method}] {index}/{len(records)} pred={prediction or '?'} ans={record.answer} "
                f"acc={summarize(methods[method])['score']:.2f}% latency={latency:.2f}s",
                flush=True,
            )

    summary = {
        "scope": f"{len(records)} MME-RealWorld-Lite position",
        "note": "Proxy ablation because local GQA images were not found.",
        "methods": {name: summarize(rows) for name, rows in methods.items()},
        "mme_accuracy_style": {
            name: score_accuracy(
                [
                    {
                        "category": args.category,
                        "task": "Perception",
                        "prediction": row["prediction"],
                        "answer": row["answer"],
                    }
                    for row in rows
                ]
            )
            for name, rows in methods.items()
        },
    }
    (output_dir / "grounding_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_table(Path(args.table_path), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare cropped patch grounding with textual coordinate grounding.")
    parser.add_argument("--data", default="data/MME-RealWorld-Lite.json")
    parser.add_argument("--model", default="models/Qwen3.5-9B")
    parser.add_argument("--output_dir", default="outputs/grounding_ablation_mme_position")
    parser.add_argument("--table_path", default="GROUNDING_ABLATION_TABLE.md")
    parser.add_argument("--category", default="position")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=12)
    parser.add_argument("--min_pixels", type=int, default=0)
    parser.add_argument("--max_pixels", type=int, default=32768)
    parser.add_argument("--use_cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
