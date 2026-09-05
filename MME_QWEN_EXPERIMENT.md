# Qwen/MME experiment notes

See [README.md](README.md) for installation, input formats, commands, scoring definitions, and recorded results.

The standalone runner supports `vanilla`, `standard_cot`, `self_cons_cot`, `standard_fot`, `vocot`, and `focot`. For FOCOT, choose `--focot_mode prompt` (default) or `--focot_mode structured` explicitly.

- Structured mode generates multiple branches and aggregates collaboration scores; optional anchor predictions change the experiment configuration.
- Prompt-mode FOCOT places the final answer after its branches. A very short generation budget may truncate the answer.
- Choose a Transformers build that supports the selected checkpoint. Legacy VolCano dependencies are separate.
- Generation cache and internal thinking are disabled by default.
- Keep model, data, prompts, token budgets, image resolution, and anchor settings in the experiment record.
- `--dry_run` returns ground-truth labels as predictions and is only a pipeline check.
- `--resume` assumes the existing predictions belong to the same experiment.
- RealWorld accuracy, original MME scores, and API judge scores are different metrics.
- The 137-sample v2 comparison in `REBUTTAL_QWEN35_MME_TABLE.md` is preliminary and does not describe a completed v3/anchor run.