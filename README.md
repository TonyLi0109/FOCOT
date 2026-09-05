# FOCOT

FOCOT is a research codebase exploring the combination of Forest-of-Thought and visually grounded, object-centric reasoning. This repository includes early implementations based on VoCoT/VolCano and standalone scripts for comparing inference strategies on MME and MME-RealWorld-Lite with Qwen vision-language models.

Start with the Qwen/MME workflow in the repository root. The `VoCoT-main/` directory preserves the earlier model, training, data processing, and evaluation code and requires a separate dependency environment.

## Features and Repository Structure

| File or directory | Purpose |
|---|---|
| `mme_qwen_experiment.py` | Data loading, inference, resumable runs, accuracy, and MME scoring |
| `api_judge_mme_table.py` | Additional evaluation of predictions through the OpenAI API |
| `grounding_ablation_mme.py` | Grounding ablations comparing image crops with textual coordinates |
| `run_*.ps1` / `show_*.ps1` | Windows experiment launchers and progress monitoring scripts |
| `VoCoT-main/model/` | VolCano and visual FoT model implementations |
| `VoCoT-main/locals/`, `config/`, `eval/`, `utils/` | Supporting code for the earlier training and evaluation workflow |
| `REBUTTAL_QWEN35_MME_TABLE.md` | Recorded full baseline results and partial-sample comparisons |
| `GROUNDING_ABLATION_TABLE.md` | Grounding ablation results and evaluation scope |

Datasets, model weights, per-sample outputs, logs, duplicate upstream downloads, submission materials, and reviewer responses are excluded from version control. Other Python utilities in the repository root are earlier analysis scripts and may require additional dependencies and local path configuration.

## Installation

Create a Python 3.11 environment for the standalone experiments:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with your hardware. Qwen3.5 checkpoints require a Transformers build that supports `qwen3_5`; this repository does not provide a fully validated, locked environment. If loading fails with an unsupported model type, check whether your Transformers installation supports the checkpoint. The root dependencies target Qwen/MME and are separate from the legacy VolCano training environment.

## Data and Models

Prepare datasets and model checkpoints that you are authorized to use, for example:

```text
data/
  MME-RealWorld-Lite.json
  images/...
models/
  Qwen3.5-9B/...
```

`--data` accepts an MME directory, JSON, JSONL, or tab-separated TXT file. Generic JSON records must contain `image`, `question`, and `answer`; optional fields include `id`, `category`, and `pair_id`. The MME-RealWorld-Lite fields `Image`, `Text`, `Answer choices`, `Ground truth`, `Category`, `Subtask`, and `Task` are also supported.

Relative image paths are resolved from the annotation directory and locations such as its `images/` and `imgs/` subdirectories. Ensure that paths point to actual images; dataset images are not distributed with the source code. `--model_path` accepts a local checkpoint path or a Hugging Face model identifier.

## Quick Start

Validate data loading and scoring without loading a model:

```powershell
python mme_qwen_experiment.py --data data/MME-RealWorld-Lite.json --method vocot --limit 8 --dry_run --output_dir outputs/dry_run
```

**`--dry_run` uses ground-truth labels as predictions. It only checks the pipeline; its scores must not be reported as model results.**

Run the VoCoT-style prompt baseline:

```powershell
python mme_qwen_experiment.py --data data/MME-RealWorld-Lite.json --model_path models/Qwen3.5-9B --method vocot --max_pixels 65536 --max_new_tokens 40 --resume --output_dir outputs/vocot
```

Run structured FOCOT:

```powershell
python mme_qwen_experiment.py --data data/MME-RealWorld-Lite.json --model_path models/Qwen3.5-9B --method focot --focot_mode structured --focot_branch_tokens 40 --max_pixels 65536 --max_new_tokens 40 --resume --output_dir outputs/focot
```

For an initial inference check, add `--limit 8`. Generation caching and Qwen internal thinking are disabled by default; enable them with `--use_cache` and `--enable_thinking`. GPU memory usage depends on the model, image resolution, and output length.

Supported methods are `vanilla`, `standard_cot`, `self_cons_cot`, `standard_fot`, `vocot`, and `focot`. In the standalone script, these names identify prompts or inference strategies; they do not imply that the original papers' trained weights are loaded. `self_cons_cot` and `standard_fot` are single-prompt baselines. `focot --focot_mode structured` generates multiple branches and applies collaborative scoring and voting. The current branch prompt includes a preference constraint concerning option E; record this setting when comparing experiments.

Use `--anchor_predictions` to introduce existing baseline predictions as an additional node and `--anchor_weight` to set its weight. Report experiments with and without anchors separately. The PowerShell launchers retain their respective experiment parameters. Before running them, check output directories, anchor inputs, and whether they automatically invoke paid API evaluation.

PowerShell launchers use `python` from PATH by default. Set the `FOCOT_PYTHON` environment variable to specify an absolute interpreter path.

## Outputs and Metrics

Each run writes `{method}_predictions.json` and `{method}_summary.json` to `--output_dir`. Use `--resume` to continue from existing predictions. Reuse an output directory only when the data, model, and inference parameters are unchanged.

Original MME scores combine per-category accuracy and paired accuracy-plus. MME-RealWorld-Lite uses answer-choice accuracy. In the script's table mapping, both `Obj.` and `Ima.` correspond to Perception, `Log.` corresponds to Reasoning, and both `Acc.` and `Total` correspond to overall accuracy. These columns are not independent object detection or image quality metrics, nor are they API judge scores.

The following results are recorded in the repository's result documents. Model inference was not rerun during repository preparation.

| Method | Evaluation scope | Accuracy (%) |
|---|---|---:|
| VOCOT | Full set of 1,919 samples | 35.44 |
| VOCOT | Same prefix of 137 samples | 29.20 |
| FOCOT v2 structured | Same prefix of 137 samples | 31.39 |

The 137-sample comparison is preliminary and contains only Perception samples. It is neither a full-benchmark FOCOT result nor a reproduction result for the current v3/anchor scripts. The grounding ablation uses 20 MME position samples as a small proxy experiment, not a full GQA evaluation. See the [experiment results](REBUTTAL_QWEN35_MME_TABLE.md) for the detailed scope.

## Optional API Evaluation

API keys are read from environment variables and should never be embedded in source code or committed to Git. Set `OPENAI_API_KEY` in your current terminal, then run:

```powershell
python api_judge_mme_table.py --method_file VOCOT=outputs/vocot/vocot_predictions.json FOCOT=outputs/focot/focot_predictions.json --judge_model YOUR_JUDGE_MODEL --output_dir outputs/api_judge --resume
```

Replace `YOUR_JUDGE_MODEL` with a model available to your account. This step sends the required evaluation text and, by default, images to the API and incurs usage charges. It is separate from local answer-matching evaluation.

## Legacy VoCoT/VolCano Implementation

See the [retained upstream README](VoCoT-main/README.md). Use `VoCoT-main/requirements.txt` in a separate environment and prepare data and weights according to the placeholder paths in `config/`. This dependency list was copied from the local `VOCOT/requirements.txt` and includes an older Transformers version; keep it separate from the root Qwen environment. The legacy training and evaluation workflows were not validated during repository preparation.

## Attribution and Licensing

This project contains modifications based on VoCoT and draws inspiration from Forest-of-Thought. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source attribution. No repository-wide open-source license is currently declared; third-party code, models, and datasets remain subject to their respective terms.
