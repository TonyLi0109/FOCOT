# FOCOT

FOCOT 是探索 Forest-of-Thought 与视觉对象中心推理结合的研究代码项目。本仓库包含基于 VoCoT/VolCano 的早期实现，以及使用 Qwen 视觉语言模型进行 MME 与 MME-RealWorld-Lite 对比实验的独立脚本。

当前推荐从根目录的 Qwen/MME 工作流开始。`VoCoT-main/` 保留早期模型、训练、数据处理和评测实现，使用独立依赖环境。

## 功能与目录

| 文件或目录 | 用途 |
|---|---|
| `mme_qwen_experiment.py` | 数据加载、推理、断点续跑、准确率和 MME 评分 |
| `api_judge_mme_table.py` | 使用 OpenAI API 对预测结果进行额外评审 |
| `grounding_ablation_mme.py` | 图像裁剪与文本坐标的 grounding 消融实验 |
| `run_*.ps1` / `show_*.ps1` | Windows 实验启动与进度查看脚本 |
| `VoCoT-main/model/` | VolCano 与视觉 FoT 模型实现 |
| `VoCoT-main/locals/`、`config/`、`eval/`、`utils/` | 早期训练和评测配套代码 |
| `REBUTTAL_QWEN35_MME_TABLE.md` | 已记录的完整基线与部分样本对比结果 |
| `GROUNDING_ABLATION_TABLE.md` | grounding 消融结果与实验范围 |

数据集、模型权重、逐样本输出、日志、重复下载的上游工程、论文投稿材料和审稿回复不纳入版本控制。根目录的其他 Python 工具属于早期分析辅助脚本，可能需要额外依赖和本地路径配置。

## 安装

建议为独立实验创建 Python 3.11 环境：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

请安装与硬件匹配的 PyTorch。Qwen3.5 检查点需要支持 `qwen3_5` 的 Transformers 构建；本仓库未提供经过完整验证的锁定环境。若加载时出现不支持模型类型的错误，请先检查所装 Transformers 对该检查点的支持。根目录依赖用于 Qwen/MME，不适用于旧版 VolCano 训练环境。

## 数据与模型

自行准备有权使用的数据和模型，例如：

```text
data/
  MME-RealWorld-Lite.json
  images/...
models/
  Qwen3.5-9B/...
```

`--data` 支持 MME 目录、JSON、JSONL 或制表符分隔的 TXT。通用 JSON 记录至少包含 `image`、`question`、`answer`；还可包含 `id`、`category`、`pair_id`。MME-RealWorld-Lite 的 `Image`、`Text`、`Answer choices`、`Ground truth`、`Category`、`Subtask` 和 `Task` 字段也受支持。

相对图像路径会从标注所在目录以及对应的 `images/`、`imgs/` 等位置查找。请保证路径指向实际图片；图片不会随源码上传。`--model_path` 可以是本地检查点路径或 Hugging Face 模型标识。

## 快速运行

先检查数据加载和评分流程，无需加载模型：

```powershell
python mme_qwen_experiment.py --data data/MME-RealWorld-Lite.json --method vocot --limit 8 --dry_run --output_dir outputs/dry_run
```

**`--dry_run` 直接使用标签作为预测，仅用于检查流程，产生的分数不能作为模型实验结果。**

运行 VoCoT 风格提示基线：

```powershell
python mme_qwen_experiment.py --data data/MME-RealWorld-Lite.json --model_path models/Qwen3.5-9B --method vocot --max_pixels 65536 --max_new_tokens 40 --resume --output_dir outputs/vocot
```

运行结构化 FOCOT：

```powershell
python mme_qwen_experiment.py --data data/MME-RealWorld-Lite.json --model_path models/Qwen3.5-9B --method focot --focot_mode structured --focot_branch_tokens 40 --max_pixels 65536 --max_new_tokens 40 --resume --output_dir outputs/focot
```

首次推理可加 `--limit 8`。默认禁用生成缓存与 Qwen 内部 thinking；可通过 `--use_cache`、`--enable_thinking` 开启。显存占用随模型、图片分辨率和输出长度变化。

支持的方法：`vanilla`、`standard_cot`、`self_cons_cot`、`standard_fot`、`vocot`、`focot`。这些名称在独立脚本中代表提示或推理策略，并不意味着加载了原论文的训练权重。`self_cons_cot` 和 `standard_fot` 是单次提示基线；`focot --focot_mode structured` 则生成多个分支，进行协作评分和投票。当前分支提示包含对选项 E 的偏好约束，比较实验时应记录该设定。

可使用 `--anchor_predictions` 引入已有基线预测作为额外节点，使用 `--anchor_weight` 设置权重。这与不使用 anchor 的实验应分别报告。已有 PowerShell 启动脚本保留各自实验参数，运行前请核对输出目录、anchor 输入和是否会自动调用付费 API 评审。

PowerShell 启动脚本默认使用 PATH 中的 `python`，也可通过 `FOCOT_PYTHON` 环境变量指定解释器绝对路径。

## 输出与指标

每次运行在 `--output_dir` 下写入 `{method}_predictions.json` 与 `{method}_summary.json`。使用 `--resume` 可继续已有预测；仅在数据、模型和推理参数一致时复用输出目录。

传统 MME 使用类别内 accuracy 与成对 accuracy-plus 计算分数。MME-RealWorld-Lite 使用选项准确率；脚本的表格映射中 `Obj.` 与 `Ima.` 都对应 Perception，`Log.` 对应 Reasoning，`Acc.` 与 `Total` 都对应总体准确率。这些列不是独立的对象检测或图像质量指标，也不是 API judge 的评分。

已有记录如下（来自仓库中的结果文档，本次整理未重跑模型）：

| 方法 | 样本范围 | Accuracy (%) |
|---|---|---:|
| VOCOT | 完整 1919 条 | 35.44 |
| VOCOT | 相同的前 137 条 | 29.20 |
| FOCOT v2 structured | 相同的前 137 条 | 31.39 |

137 条对比是仅含 Perception 的初步结果，不能作为 FOCOT 全集成绩，也不能视为当前 v3/anchor 脚本的复现结果。Grounding 消融使用 20 条 MME position 样本，属于小规模代理实验，并非完整 GQA 评测。详细范围见 [实验结果](REBUTTAL_QWEN35_MME_TABLE.md)。

## 可选 API 评审

API 密钥仅从环境变量读取，不应写入源码或提交到 Git。运行前在当前终端配置 `OPENAI_API_KEY`，再执行：

```powershell
python api_judge_mme_table.py --method_file VOCOT=outputs/vocot/vocot_predictions.json FOCOT=outputs/focot/focot_predictions.json --judge_model YOUR_JUDGE_MODEL --output_dir outputs/api_judge --resume
```

将 `YOUR_JUDGE_MODEL` 替换为账号可访问的模型。此步骤会把评审所需文本及默认启用的图片发送至 API，并产生费用。它独立于本地答案匹配评分。

## 早期 VoCoT/VolCano 实现

参见 [保留的上游 README](VoCoT-main/README.md)。在独立环境中使用 `VoCoT-main/requirements.txt`，根据 `config/` 中的占位路径准备数据与权重。该依赖清单来自本地 `VOCOT/requirements.txt`，包含旧版 Transformers，不要与根目录 Qwen 环境混装。早期训练和评测尚未在本次仓库整理中验证。

## 来源与许可

本项目包含基于 VoCoT 的代码改动，并受 Forest-of-Thought 启发。来源说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。当前未为整个项目声明统一开源许可证；第三方代码、模型和数据仍遵循各自原有条款。
