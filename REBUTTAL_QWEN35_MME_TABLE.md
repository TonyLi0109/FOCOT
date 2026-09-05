# Qwen-3.5-9B MME-RealWorld-Lite Results

## Full Completed Run

| Method | Scope | Obj. | Log. | Ima. | Acc. | Total |
|---|---:|---:|---:|---:|---:|---:|
| VOCOT | 1919 / 1919 | 36.18 | 34.27 | 36.18 | 35.44 | 35.44 |

## Preliminary Prefix Comparison

The following table compares VOCOT and FOCOT on the exact same completed prefix of MME-RealWorld-Lite. This prefix currently contains Perception-only samples, so Log. is not reported.

| Method | Scope | Obj. | Log. | Ima. | Acc. | Total |
|---|---:|---:|---:|---:|---:|---:|
| VOCOT | 137 / 1919 prefix | 29.20 | N/A | 29.20 | 29.20 | 29.20 |
| FOCOT v2 structured | 137 / 1919 prefix | 31.39 | N/A | 31.39 | 31.39 | 31.39 |

## Grounding Mechanism Ablation

This run uses a 20-sample MME-RealWorld-Lite `position` subset as a compact proxy because local GQA images are not available. The comparison keeps the backbone, decoding settings, and evaluated samples fixed.

| Grounding Mechanism | GQA Total Score | Avg. Latency (s/query) | Peak GPU Memory (GB) |
|---|---:|---:|---:|
| Cropped Image Patches | 15.00* | 16.886 | 9.518 |
| Textual Coordinates (Ours) | 40.00* | 18.712 | 9.520 |

\* Proxy score on 20 MME-RealWorld-Lite position samples, not full GQA. For the rebuttal, use this as a compact grounding ablation or rename the score column to `MME Position Score` unless the GQA images are supplied and rerun.

## Prefix Error Complementarity

| Metric | Count |
|---|---:|
| Both correct | 27 |
| FOCOT-only correct | 16 |
| VOCOT-only correct | 13 |

## Rebuttal-Safe Sentence

On the currently completed MME-RealWorld-Lite prefix under the Qwen-3.5-9B backbone, structured FOCOT improves over VOCOT from 29.20% to 31.39% accuracy on the same 137 evaluated samples. This partial result suggests that the proposed multi-branch collaboration can recover additional cases beyond VOCOT, with 16 FOCOT-only correct predictions versus 13 VOCOT-only correct predictions. We will report the full benchmark score after the remaining samples finish.
