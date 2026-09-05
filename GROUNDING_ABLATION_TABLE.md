# Grounding Mechanism Ablation

Current run: 20-sample MME-RealWorld-Lite `position` subset proxy, because local GQA images are not present. The comparison keeps the backbone, decoding settings, and evaluated samples fixed.

| Grounding Mechanism | GQA Total Score | Avg. Latency (s/query) | Peak GPU Memory (GB) |
|---|---:|---:|---:|
| Cropped Image Patches | 15.00* | 16.886 | 9.518 |
| Textual Coordinates (Ours) | 40.00* | 18.712 | 9.520 |

\* Proxy score on 20 MME-RealWorld-Lite position samples, not full GQA. Replace the column label with `MME Position Score` unless GQA images are supplied and rerun.
