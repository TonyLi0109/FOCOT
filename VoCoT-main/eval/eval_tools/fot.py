# -*- coding: utf-8 -*-
import json
import os
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
import math
import re
from typing import Any, Dict, Tuple
from contextlib import nullcontext

from locals.datasets import SFT_DataCollator
from constants import ALL_IMG_TOKENS_STR, DEFAULT_GRD_TOKEN

GQA_ACTIVATION = 'Please answer the question based on the image. Provide a clear and direct text answer.'
MAX_INPUT_LEN = 4096
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.1

# -------------------- 基础工具 --------------------
def _get_device_of_model(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')

def _to_device(obj, device: torch.device):
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).to(device, non_blocking=True)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj

def _ensure_dtype(batch: dict):
    try:
        if 'input_ids' in batch and isinstance(batch['input_ids'], torch.Tensor):
            if batch['input_ids'].dtype != torch.long:
                batch['input_ids'] = batch['input_ids'].long()
        if 'attention_mask' in batch and isinstance(batch['attention_mask'], torch.Tensor):
            if batch['attention_mask'].dtype != torch.long:
                batch['attention_mask'] = batch['attention_mask'].long()
    except RuntimeError as e:
        if "device-side assert" in str(e).lower():
            print(f"Warning: CUDA assert in dtype conversion, skipping some conversions")
        else:
            raise e
    # 视觉字段
    for k in ['pixel_values', 'images', 'image_tensors', 'vision_indices', 'vision_tokens']:
        if k in batch:
            try:
                v = batch[k]
                if isinstance(v, torch.Tensor):
                    if k in ['vision_indices', 'vision_tokens']:
                        if v.dtype != torch.long:
                            batch[k] = v.long()
                    else:
                        if not torch.is_floating_point(v):
                            batch[k] = v.float()
                elif isinstance(v, (list, tuple)):
                    if v and isinstance(v[0], torch.Tensor):
                        if k in ['vision_indices', 'vision_tokens']:
                            v = [x.long() if x.dtype != torch.long else x for x in v]
                        else:
                            v = [x.float() if not torch.is_floating_point(x) else x for x in v]
                        try:
                            batch[k] = torch.stack(v, dim=0)
                        except Exception:
                            # 保持 list 形态也行，后续 _to_device 会处理
                            batch[k] = v
            except RuntimeError as e:
                if "device-side assert" in str(e).lower():
                    print(f"Warning: CUDA assert in {k} processing, removing from batch")
                    batch.pop(k, None)
                else:
                    raise e
    return batch

def _has_nan_inf(t: torch.Tensor) -> bool:
    if not torch.is_tensor(t):
        return False
    return torch.isnan(t).any().item() or torch.isinf(t).any().item()

def _range_str(t: torch.Tensor) -> str:
    if t.numel() == 0 or not torch.is_tensor(t):
        return "empty"
    try:
        return f"[min={t.min().item()}, max={t.max().item()}]"
    except Exception:
        return "n/a"

def _print_tensor_info(name: str, t: torch.Tensor):
    if not isinstance(t, torch.Tensor):
        print(f"  - {name}: <non-tensor> type={type(t)}")
        return
    print(f"  - {name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, range={_range_str(t)}, nan/inf={_has_nan_inf(t)}")

def initialize_model_attributes(model):
    try:
        if not hasattr(model, 'boi_token'):
            model.boi_token = getattr(model.tokenizer, 'boi_token', '<image>')
        if not hasattr(model, 'eoi_token'):
            model.eoi_token = getattr(model.tokenizer, 'eoi_token', '</image>')
        if not hasattr(model, 'cache_images'):
            model.cache_images = False
        if not hasattr(model, 'image_cache'):
            model.image_cache = {}
        if not hasattr(model, 'cached_features'):
            model.cached_features = {}
        model.eval()
    except Exception as e:
        print(f"Warning: Failed to initialize some model attributes: {e}")

def validate_and_fix_tokens(input_ids, tokenizer, max_length=MAX_INPUT_LEN):
    if input_ids is None:
        return None
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
    if input_ids.dtype != torch.long:
        input_ids = input_ids.long()

    # 长度裁剪
    if input_ids.shape[-1] > max_length:
        input_ids = input_ids[..., :max_length]

    # 词表范围
    try:
        vocab_size = len(tokenizer)
    except Exception:
        vocab_size = getattr(tokenizer, 'vocab_size', None)
    if vocab_size is None:
        try:
            # 安全地获取最大值，避免CUDA assert
            max_val = input_ids.max()
            if torch.cuda.is_available() and input_ids.is_cuda:
                torch.cuda.synchronize()  # 确保操作完成
            vocab_size = int(max(0, max_val.item())) + 1
        except RuntimeError as e:
            if "device-side assert" in str(e).lower():
                # 如果遇到CUDA assert，使用默认词表大小
                vocab_size = getattr(tokenizer, 'vocab_size', 50000)
                print(f"Warning: CUDA assert in vocab size calculation, using default: {vocab_size}")
            else:
                raise e
    
    # 安全地检查无效token
    try:
        invalid_mask = (input_ids < 0) | (input_ids >= vocab_size)
        if invalid_mask.any():
            unk_id = getattr(tokenizer, 'unk_token_id', None)
            if unk_id is None:
                unk_id = getattr(tokenizer, 'pad_token_id', 0)
            input_ids = input_ids.clone()
            input_ids[invalid_mask] = int(unk_id)
    except RuntimeError as e:
        if "device-side assert" in str(e).lower():
            print(f"Warning: CUDA assert in token validation, skipping validation")
        else:
            raise e

    # 清理 NaN/Inf（理论上不会出现在 long，但以防万一）
    return input_ids

def sanitize_batch(batch, tokenizer, device) -> Dict[str, torch.Tensor]:
    if batch is None:
        return None
    
    try:
        batch = _ensure_dtype(batch)
        batch = _to_device(batch, device)

        if 'input_ids' in batch and isinstance(batch['input_ids'], torch.Tensor):
            input_ids = batch['input_ids']
            
            # 检查批处理大小，限制为1以避免内存问题
            if input_ids.shape[0] > 1:
                print(f"Warning: batch size {input_ids.shape[0]} > 1, taking first sample only")
                input_ids = input_ids[:1]
                batch['input_ids'] = input_ids
                # 同时调整其他相关tensor
                for key in ['attention_mask', 'labels']:
                    if key in batch and isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key][:1]
            
            # 检查序列长度
            seq_len = input_ids.shape[1] if len(input_ids.shape) > 1 else input_ids.shape[0]
            if seq_len > MAX_INPUT_LEN:
                print(f"Warning: sequence length {seq_len} exceeds MAX_INPUT_LEN {MAX_INPUT_LEN}, truncating")
                input_ids = input_ids[:, :MAX_INPUT_LEN] if len(input_ids.shape) > 1 else input_ids[:MAX_INPUT_LEN]
                batch['input_ids'] = input_ids
            
            batch['input_ids'] = validate_and_fix_tokens(batch['input_ids'], tokenizer, MAX_INPUT_LEN)
            batch['input_ids'] = batch['input_ids'].to(device, non_blocking=True)

        if 'attention_mask' not in batch and 'input_ids' in batch:
            batch['attention_mask'] = torch.ones_like(batch['input_ids'], dtype=torch.long, device=device)
        if 'attention_mask' in batch and 'input_ids' in batch:
            if batch['attention_mask'].shape != batch['input_ids'].shape:
                batch['attention_mask'] = torch.ones_like(batch['input_ids'], dtype=torch.long, device=device)
            # 二值化
            batch['attention_mask'] = (batch['attention_mask'] > 0).long()
    
    except Exception as e:
        print(f"Error in sanitize_batch: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    # 视觉索引（若存在）强制非负
    for k in ['vision_indices', 'vision_tokens']:
        if k in batch and isinstance(batch[k], torch.Tensor):
            try:
                v = batch[k]
                if v.dtype != torch.long:
                    v = v.long()
                v = torch.clamp(v, min=0)
                batch[k] = v.to(device, non_blocking=True)
            except RuntimeError as e:
                if "device-side assert" in str(e).lower():
                    print(f"Warning: CUDA assert in {k} processing, removing from batch")
                    batch.pop(k, None)
                else:
                    raise e

    # 清理 NaN/Inf
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v) and _has_nan_inf(v):
            v = torch.nan_to_num(v, nan=0.0, posinf=1e4, neginf=-1e4)
            batch[k] = v

    return batch

def _log_batch(batch: Dict[str, Any], title="Batch snapshot"):
    print(f"\n== {title} ==")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            _print_tensor_info(k, v)
        else:
            print(f"  - {k}: type={type(v)}")
    print("== End snapshot ==\n")

def _cuda_sync():
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

def _is_cuda_assert(e: Exception) -> bool:
    msg = str(e).lower()
    return ("device-side assert" in msg) or ("device side assert" in msg)

def _move_model(model, device: torch.device):
    current = _get_device_of_model(model)
    if current != device:
        model.to(device)

# -------------------- 生成（含 CPU 回退） --------------------
def _generate_once(model, **kwargs):
    """安全的单次生成函数，包含CUDA错误预防机制"""
    try:
        # 确保所有tensor在同一设备上
        device = _get_device_of_model(model)
        
        # 验证关键参数
        if 'input_ids' not in kwargs:
            raise ValueError("Missing required parameter: input_ids")
        
        input_ids = kwargs['input_ids']
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("input_ids must be a torch.Tensor")
        
        # 确保input_ids在正确设备上
        if input_ids.device != device:
            kwargs['input_ids'] = input_ids.to(device)
        
        # 确保attention_mask也在正确设备上
        if 'attention_mask' in kwargs and isinstance(kwargs['attention_mask'], torch.Tensor):
            if kwargs['attention_mask'].device != device:
                kwargs['attention_mask'] = kwargs['attention_mask'].to(device)
        
        # 设置安全的生成参数
        safe_kwargs = kwargs.copy()
        safe_kwargs.setdefault('max_new_tokens', min(kwargs.get('max_new_tokens', DEFAULT_MAX_NEW_TOKENS), 256))
        safe_kwargs.setdefault('do_sample', False)  # 使用贪心解码避免随机性
        safe_kwargs.setdefault('use_cache', False)  # 禁用缓存减少内存使用
        safe_kwargs.setdefault('pad_token_id', model.tokenizer.pad_token_id)
        safe_kwargs.setdefault('eos_token_id', model.tokenizer.eos_token_id)
        
        with torch.inference_mode():
            return model.generate(**safe_kwargs)
            
    except Exception as e:
        print(f"Error in _generate_once: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e

def safe_generate(model, **kwargs):
    """先在当前 device 试；如果遇到 device-side assert，自动切到 CPU 重试"""
    device = _get_device_of_model(model)

    # 关闭 autocast，避免半精度奇葩错误 - 修复上下文管理器问题
    try:
        if torch.cuda.is_available():
            autocast_cuda_disabled = torch.autocast("cuda", enabled=False)
        else:
            autocast_cuda_disabled = nullcontext()
    except Exception:
        autocast_cuda_disabled = nullcontext()
    try:
        # 统一张量到同一设备并校验
        tmp = {}
        if 'input_ids' in kwargs:
            tmp['input_ids'] = kwargs['input_ids']
        if 'attention_mask' in kwargs:
            tmp['attention_mask'] = kwargs['attention_mask']
        tmp = sanitize_batch(tmp, model.tokenizer, device)

        if tmp is None:
            return None
        kwargs['input_ids'] = tmp['input_ids']
        kwargs['attention_mask'] = tmp['attention_mask']

        # 其它 tensor 参数也搬到 device
        for k, v in list(kwargs.items()):
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.to(device, non_blocking=True)

        gen_kwargs = {
            'max_new_tokens': min(kwargs.pop('max_new_tokens', DEFAULT_MAX_NEW_TOKENS), 512),
            'temperature': max(kwargs.pop('temperature', DEFAULT_TEMPERATURE), 0.0),
            'do_sample': kwargs.pop('do_sample', DEFAULT_TEMPERATURE > 0),
            'pad_token_id': model.tokenizer.pad_token_id,
            'eos_token_id': model.tokenizer.eos_token_id,
            'use_cache': False,
        }
        kwargs.update(gen_kwargs)

        _move_model(model, device)
        _cuda_sync()

        

        try:
            # 在调用_generate_once前进行额外的输入验证
            input_ids = kwargs.get('input_ids')
            if input_ids is not None:
                # 检查input_ids的形状和值范围
                if len(input_ids.shape) != 2:
                    print(f"Warning: input_ids shape is {input_ids.shape}, expected 2D tensor")
                    return None
                
                # 检查序列长度
                seq_len = input_ids.shape[1]
                if seq_len > MAX_INPUT_LEN:
                    print(f"Warning: sequence length {seq_len} exceeds MAX_INPUT_LEN {MAX_INPUT_LEN}")
                    # 截断序列
                    input_ids = input_ids[:, :MAX_INPUT_LEN]
                    kwargs['input_ids'] = input_ids
                    if 'attention_mask' in kwargs:
                        kwargs['attention_mask'] = kwargs['attention_mask'][:, :MAX_INPUT_LEN]
                
                # 检查token值范围
                vocab_size = getattr(model.tokenizer, 'vocab_size', 50000)
                invalid_tokens = (input_ids >= vocab_size) | (input_ids < 0)
                if invalid_tokens.any():
                    print(f"Warning: Found {invalid_tokens.sum().item()} invalid tokens")
                    # 将无效token替换为pad_token_id
                    input_ids[invalid_tokens] = model.tokenizer.pad_token_id
                    kwargs['input_ids'] = input_ids
            
            with autocast_cuda_disabled:
                out = _generate_once(model, **kwargs)
            _cuda_sync()
            return out
        except RuntimeError as e:
            _cuda_sync()
            print("Generation error on device", device, ":", e)
            if _is_cuda_assert(e):
                print(">>> CUDA assert detected. Fallback to CPU and retry once.")
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                cpu = torch.device('cpu')
                # 把一切切到 CPU
                _move_model(model, cpu)
                kwargs['input_ids'] = kwargs['input_ids'].to(cpu)
                kwargs['attention_mask'] = kwargs['attention_mask'].to(cpu)
                for k, v in list(kwargs.items()):
                    if isinstance(v, torch.Tensor):
                        kwargs[k] = v.to(cpu)
                try:
                    out = _generate_once(model, **kwargs)
                    return out
                except Exception as e2:
                    print("CPU retry also failed:", e2)
                    return None
            return None
    except Exception as e:
        print("Generation error:", e)
        return None

# -------------------- 逐步推理 --------------------
def truncate_middle_tokens(input_ids, max_length=MAX_INPUT_LEN, keep_start=512, keep_end=512):
    """
    分段推理策略：当序列过长时，保留开头和结尾的重要部分，截断中间内容
    """
    if input_ids.shape[1] <= max_length:
        return input_ids
    
    # 确保保留的开头和结尾不超过总长度
    total_keep = keep_start + keep_end
    if total_keep >= max_length:
        keep_start = max_length // 2
        keep_end = max_length - keep_start
    
    # 保留开头和结尾部分
    start_part = input_ids[:, :keep_start]
    end_part = input_ids[:, -keep_end:]
    
    # 拼接开头和结尾
    truncated = torch.cat([start_part, end_part], dim=1)
    print(f"[OPTIMIZATION] Sequence truncated from {input_ids.shape[1]} to {truncated.shape[1]} tokens")
    return truncated

def step_by_step_inference(model, batch, max_steps=1, max_new_tokens=50, temperature=0.1):
    """改进版本：每个分析维度生成多个候选思维"""
    print(f"[DEBUG] Starting enhanced step_by_step_inference with max_steps={max_steps}")
    
    try:
        initialize_model_attributes(model)
        device = _get_device_of_model(model)
        print(f"[DEBUG] Model device: {device}")
        
        if batch is None:
            print("[ERROR] Batch is None")
            return "Error: Batch is None", "Error", []
        
        # 安全的batch处理
        try:
            batch = sanitize_batch(batch, model.tokenizer, device)
            if batch is None or 'input_ids' not in batch:
                print("[ERROR] Batch sanitization failed")
                return "Error: Batch sanitization failed", "Error", []
        except Exception as e:
            print(f"[ERROR] Batch sanitization error: {e}")
            return f"Error: Batch sanitization - {e}", "Error", []

        thoughts = []
        current_prompt = batch['input_ids'].clone().to(device, non_blocking=True)
        if current_prompt.dtype != torch.long:
            current_prompt = current_prompt.long()
        
        # 定义三个不同的分析维度，每个维度包含3个子角度
        analysis_dimensions = [
            {
                "name": "Visual Element Analysis",
                "sub_aspects": [
                    {
                        "name": "Object Identification",
                        "prompt": "\nStep {step} - Object Identification: Focus ONLY on identifying specific objects, items, and entities visible in the image. Answer in English only. What concrete things can I see? Avoid abstract concepts like dreams, imagination, or fantasy.",
                        "keywords": ['object', 'item', 'thing', 'entity', 'see', 'identify']
                    },
                    {
                        "name": "Scene Environment", 
                        "prompt": "\nStep {step} - Scene Environment: Analyze ONLY the background, setting, location, and environmental context visible in the image. Answer in English only. Where is this taking place? Focus on observable elements only.",
                        "keywords": ['background', 'setting', 'location', 'environment', 'scene', 'place']
                    },
                    {
                        "name": "Visual Features",
                        "prompt": "\nStep {step} - Visual Features: Examine ONLY colors, lighting, composition, and visual characteristics visible in the image. Answer in English only. How does it look? Describe only what you can see.",
                        "keywords": ['color', 'light', 'bright', 'dark', 'visual', 'appearance']
                    }
                ]
            },
            {
                "name": "Question Understanding Analysis",
                "sub_aspects": [
                    {
                        "name": "Question Focus",
                        "prompt": "\nStep {step} - Question Focus: Identify the main focus and intent of the question. Answer in English only. What is the question specifically asking about? Focus on concrete, observable elements only.",
                        "keywords": ['question', 'asking', 'focus', 'main', 'what', 'how', 'where', 'when']
                    },
                    {
                        "name": "Keyword Analysis",
                        "prompt": "\nStep {step} - Keyword Analysis: Analyze ONLY key terms related to visible objects and elements in the image. Answer in English only. Focus strictly on concrete and observable elements. Forbidden: dreams, imagination, fantasy, emotions.",
                        "keywords": ['keyword', 'term', 'important', 'key', 'word', 'concept']
                    },
                    {
                        "name": "Answer Type",
                        "prompt": "\nStep {step} - Answer Type: Determine what type of answer is expected based on the question. Answer in English only. Is this asking for yes/no, counting, description, or identification? Focus on factual responses only.",
                        "keywords": ['answer', 'type', 'format', 'yes', 'no', 'count', 'describe']
                    }
                ]
            },
            {
                "name": "Logical Reasoning Analysis",
                "sub_aspects": [
                    {
                        "name": "Direct Evidence",
                        "prompt": "\nStep {step} - Direct Evidence: Identify ONLY direct visual evidence in the image that answers the question. Answer in English only. What can I directly observe that relates to the question? No speculation or imagination.",
                        "keywords": ['evidence', 'proof', 'direct', 'observe', 'see', 'visible']
                    },
                    {
                        "name": "Reasoning Connection",
                        "prompt": "\nStep {step} - Reasoning Connection: Connect the visual evidence to the question logically. Answer in English only. How does what I see in the image relate to what the question is asking? Base reasoning only on visible elements.",
                        "keywords": ['connect', 'relate', 'link', 'because', 'therefore', 'logic']
                    },
                    {
                        "name": "Conclusion Verification",
                        "prompt": "\nStep {step} - Conclusion Verification: Verify the conclusion based on visible evidence. Answer in English only. Does my answer match what I can actually see in the image? Ensure factual accuracy only.",
                        "keywords": ['verify', 'check', 'confirm', 'accurate', 'correct', 'match']
                    }
                ]
            }
        ]
        
        # 节点质量评估函数（针对特定子角度）
        def evaluate_thought_quality_by_sub_aspect(thought_text, sub_aspect):
            """根据特定子角度评估思维节点的质量"""
            if not thought_text or len(thought_text.strip()) < 10:
                return 0.0
            
            quality_score = 0.0
            
            # 长度合理性 (10-100字符较好)
            length = len(thought_text.strip())
            if 10 <= length <= 100:
                quality_score += 0.3
            elif length > 100:
                quality_score += 0.1
            
            # 子角度特定关键词奖励
            text_lower = thought_text.lower()
            keyword_count = sum(1 for kw in sub_aspect['keywords'] if kw in text_lower)
            quality_score += min(keyword_count * 0.2, 0.4)
            
            # 通用质量指标
            if any(kw in text_lower for kw in ['image', 'picture']):
                quality_score += 0.1
            if len(thought_text.split()) >= 5:  # 至少5个词
                quality_score += 0.2
            
            return min(quality_score, 1.0)
        
        # 多候选节点生成和选择
        for step in range(max_steps):
            print(f"[DEBUG] Starting enhanced step {step + 1}/{max_steps}")
            
            try:
                # 为每个维度生成多个候选
                all_candidates = []
                
                for dim_idx, dimension in enumerate(analysis_dimensions):
                    print(f"[DEBUG] Dimension {dim_idx + 1} ({dimension['name']}):")
                    dimension_candidates = []
                    
                    # 为每个子角度生成一个候选
                    for sub_idx, sub_aspect in enumerate(dimension['sub_aspects']):
                        try:
                            step_text = sub_aspect['prompt'].format(step=step + 1)
                            
                            step_tokens = model.tokenizer.encode(step_text, add_special_tokens=False, return_tensors='pt')
                            if step_tokens is None or step_tokens.numel() == 0:
                                continue
                            step_tokens = step_tokens.to(device, non_blocking=True).long()
                            
                            # 构建当前批次
                            current_batch = {
                                'input_ids': torch.cat([current_prompt, step_tokens], dim=1),
                                'attention_mask': torch.ones(current_prompt.shape[0], current_prompt.shape[1] + step_tokens.shape[1], 
                                                            dtype=torch.long, device=device)
                            }
                            
                            # 使用稍微不同的temperature为每个子角度增加多样性
                            temp = temperature + sub_idx * 0.03
                            outputs = safe_generate(
                                model,
                                input_ids=current_batch['input_ids'],
                                attention_mask=current_batch['attention_mask'],
                                max_new_tokens=max_new_tokens,
                                temperature=temp,
                                do_sample=temp > 0,
                                pad_token_id=model.tokenizer.pad_token_id,
                                eos_token_id=model.tokenizer.eos_token_id
                            )
                            
                            if outputs is not None and len(outputs) > 0:
                                new_tokens = outputs[0][current_batch['input_ids'].shape[1]:]
                                if new_tokens.numel() > 0:
                                    step_thought = model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                                    step_thought = re.sub(r'<[^>]+>', '', step_thought).strip()
                                    
                                    # 取第一句话作为思维节点
                                    sentences = [s.strip() for s in step_thought.split('.') if s.strip()]
                                    if sentences:
                                        step_thought = sentences[0]
                                        quality = evaluate_thought_quality_by_sub_aspect(step_thought, sub_aspect)
                                        candidate = {
                                            'text': step_thought,
                                            'quality': quality,
                                            'dimension': dimension['name'],
                                            'sub_aspect': sub_aspect['name'],
                                            'dim_idx': dim_idx,
                                            'sub_idx': sub_idx
                                        }
                                        dimension_candidates.append(candidate)
                                        all_candidates.append(candidate)
                                        print(f"[DEBUG] Candidate {sub_idx + 1} ({sub_aspect['name']}): '{step_thought}' (quality: {quality:.2f})")
                            
                            # 清理内存
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            # 安全删除变量
                            if 'current_batch' in locals():
                                del current_batch
                            if 'outputs' in locals():
                                del outputs
                            if 'new_tokens' in locals():
                                del new_tokens
                            
                        except Exception as e:
                            print(f"[WARNING] Sub-aspect {sub_idx + 1} ({sub_aspect['name']}) generation failed: {e}")
                            continue
                    
                    # 显示该维度的所有候选
                    if dimension_candidates:
                        print(f"[DEBUG] {dimension['name']} - Generated {len(dimension_candidates)} candidates")
                
                # 选择最佳候选节点
                if all_candidates:
                    # 按质量排序
                    all_candidates.sort(key=lambda x: x['quality'], reverse=True)
                    
                    # 选择质量最高的候选
                    best_candidate = all_candidates[0]
                    best_thought = best_candidate['text']
                    best_quality = best_candidate['quality']
                    best_dimension = best_candidate['dimension']
                    best_sub_aspect = best_candidate['sub_aspect']
                    
                    print(f"[DEBUG] Selected best thought from {best_dimension} - {best_sub_aspect}: '{best_thought}' (quality: {best_quality:.2f})")
                    
                    if best_quality > 0.3:  # 质量阈值
                        thoughts.append(f"Step {step + 1} - {best_dimension} ({best_sub_aspect}): {best_thought}")
                        
                        # 更新current_prompt
                        step_simple = f" {best_thought}."
                        step_simple_tokens = model.tokenizer.encode(step_simple, add_special_tokens=False, return_tensors='pt')
                        if step_simple_tokens is not None and step_simple_tokens.numel() > 0:
                            step_simple_tokens = step_simple_tokens.to(device, non_blocking=True).long()
                            current_prompt = torch.cat([current_prompt, step_simple_tokens], dim=1)
                            print(f"[DEBUG] Updated current_prompt to {current_prompt.shape[1]} tokens")
                    else:
                        print(f"[WARNING] Step {step + 1} best candidate quality too low: {best_quality:.2f}")
                else:
                    print(f"[WARNING] Step {step + 1} no valid candidates generated")
                    
            except Exception as e:
                print(f"[ERROR] Step {step + 1} failed completely: {e}")
                break
        
        # 改进的最终答案生成
        final_answer = ""
        print(f"[DEBUG] Starting enhanced final answer generation")
        
        try:
            # 从batch中提取原始问题文本
            original_question = ""
            if 'input_ids' in batch:
                # 解码input_ids来获取原始问题
                original_text = model.tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
                # 提取问题部分（通常在最后）
                if '?' in original_text:
                    question_parts = original_text.split('?')
                    if len(question_parts) > 1:
                        # 取最后一个问号前的内容作为问题
                        original_question = question_parts[-2].split('.')[-1].strip() + '?'
            
            # 使用更明确的最终答案prompt，直接引用原始问题
            if original_question:
                final_prompt_text = f"\nNow, based on my step-by-step analysis above, I will directly answer the question: '{original_question}'\nMy answer is: "
            else:
                final_prompt_text = "\nNow, based on my step-by-step analysis above, I will directly answer the original question.\nMy answer is: "
            
            final_tokens = model.tokenizer.encode(final_prompt_text, add_special_tokens=False, return_tensors='pt')
            
            if final_tokens is not None and final_tokens.numel() > 0:
                final_tokens = final_tokens.to(device, non_blocking=True).long()
                final_batch = {
                    'input_ids': torch.cat([current_prompt, final_tokens], dim=1),
                    'attention_mask': torch.ones(current_prompt.shape[0], current_prompt.shape[1] + final_tokens.shape[1], 
                                                dtype=torch.long, device=device)
                }
                
                final_outputs = safe_generate(
                    model,
                    input_ids=final_batch['input_ids'],
                    attention_mask=final_batch['attention_mask'],
                    max_new_tokens=min(max_new_tokens, 50),  # 增加到50个token以允许更完整的答案
                    temperature=0.0,  # 使用确定性生成
                    do_sample=False,
                    pad_token_id=model.tokenizer.pad_token_id,
                    eos_token_id=model.tokenizer.eos_token_id
                )
                
                if final_outputs is not None and len(final_outputs) > 0:
                    final_new_tokens = final_outputs[0][final_batch['input_ids'].shape[1]:]
                    if final_new_tokens.numel() > 0:
                        final_answer = model.tokenizer.decode(final_new_tokens, skip_special_tokens=True).strip()
                        final_answer = re.sub(r'<[^>]+>', '', final_answer).strip()
                        
                        # 更智能的答案提取
                        sentences = [s.strip() for s in final_answer.split('.') if s.strip()]
                        if sentences:
                            # 取第一个完整句子作为答案
                            final_answer = sentences[0]
                            # 如果答案太短，尝试取前两个句子
                            if len(final_answer.split()) < 3 and len(sentences) > 1:
                                final_answer = sentences[0] + '. ' + sentences[1]
                        
                        print(f"[DEBUG] Generated final answer: '{final_answer}'")
        except Exception as e:
            print(f"[ERROR] Final answer generation failed: {e}")
        
            # 如果没有生成最终答案，从思维中提取
            if not final_answer and thoughts:
                try:
                    # 从最后一个思维中提取答案
                    last_thought = thoughts[-1]
                    if ': ' in last_thought:
                        final_answer = last_thought.split(': ', 1)[1]
                    else:
                        final_answer = "Unable to determine"
                    print(f"[DEBUG] Extracted answer from thoughts: '{final_answer}'")
                except Exception as e:
                    print(f"[ERROR] Failed to extract answer from thoughts: {e}")
                    final_answer = "Unable to determine"
        
        full_reasoning = ("\n".join(thoughts) + f"\nFinal Answer: {final_answer}") if thoughts else f"Direct Answer: {final_answer}"
        
        # 最终清理
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[DEBUG] Enhanced inference completed")
            print(f"[DEBUG] Total thoughts generated: {len(thoughts)}")
            print(f"[DEBUG] Final answer: '{final_answer}'")
        except Exception as e:
            print(f"[ERROR] Final cleanup failed: {e}")
        
        # 构建详细的返回信息
        detailed_result = {
            'full_reasoning': full_reasoning,
            'final_answer': final_answer,
            'raw_prediction': final_answer,  # 添加原始预测
            'cleaned_prediction': clean_prediction(final_answer),
            'thoughts': thoughts,
            'reasoning_steps': len(thoughts),
            'inference_success': bool(final_answer)
        }
        
        return detailed_result, final_answer, thoughts  # 返回原始final_answer
        
    except Exception as e:
        print(f"[ERROR] Enhanced step_by_step_inference failed: {e}")
        return f"Error: {e}", "Error", []

def distributed_step_by_step_inference(model, batch, max_steps=3, max_new_tokens=50, temperature=0.1):
    """分布式推理：先思考第一步，再逐步推理"""
    print(f"[DEBUG] Starting distributed step-by-step inference with max_steps={max_steps}")
    
    try:
        initialize_model_attributes(model)
        device = _get_device_of_model(model)
        
        if batch is None:
            return "Error: Batch is None", "Error", []
        
        batch = sanitize_batch(batch, model.tokenizer, device)
        if batch is None or 'input_ids' not in batch:
            return "Error: Batch sanitization failed", "Error", []

        thoughts = []
        current_prompt = batch['input_ids'].clone().to(device, non_blocking=True)
        if current_prompt.dtype != torch.long:
            current_prompt = current_prompt.long()
        
        # 分布式推理的结构化prompt模板
        distributed_prompts = [
            "\nFirst, let me think about what I can observe in this image. I need to carefully identify all the key visual elements, objects, people, and their spatial relationships before proceeding to answer the question.",
            "\nNow that I have analyzed the visual content, let me understand what the question is specifically asking. I need to connect the question requirements with the visual elements I identified in the previous step.",
            "\nBased on my visual analysis and question understanding, I can now apply logical reasoning to determine the correct answer. Let me think through this step by step using the evidence from the image."
        ]
        
        # 第一步：独立思考阶段
        print("[DEBUG] Phase 1: Independent thinking")
        first_step_prompt = distributed_prompts[0]
        
        try:
            step_tokens = model.tokenizer.encode(first_step_prompt, add_special_tokens=False, return_tensors='pt')
            step_tokens = step_tokens.to(device, non_blocking=True).long()
            
            first_batch = {
                'input_ids': torch.cat([current_prompt, step_tokens], dim=1),
                'attention_mask': torch.ones(current_prompt.shape[0], current_prompt.shape[1] + step_tokens.shape[1], 
                                            dtype=torch.long, device=device)
            }
            
            # 生成第一步思考
            outputs = safe_generate(
                model,
                input_ids=first_batch['input_ids'],
                attention_mask=first_batch['attention_mask'],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id
            )
            
            if outputs is not None and len(outputs) > 0:
                new_tokens = outputs[0][first_batch['input_ids'].shape[1]:]
                if new_tokens.numel() > 0:
                    first_thought = model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    first_thought = re.sub(r'<[^>]+>', '', first_thought).strip()
                    
                    # 提取核心思考内容
                    sentences = [s.strip() for s in first_thought.split('.') if s.strip()]
                    if sentences:
                        first_thought = '. '.join(sentences[:2])  # 取前两句
                        thoughts.append(f"Step 1 - Visual Analysis: {first_thought}")
                        
                        # 更新prompt以包含第一步思考
                        step_simple = f" {first_thought}."
                        step_simple_tokens = model.tokenizer.encode(step_simple, add_special_tokens=False, return_tensors='pt')
                        step_simple_tokens = step_simple_tokens.to(device, non_blocking=True).long()
                        current_prompt = torch.cat([current_prompt, step_simple_tokens], dim=1)
                        
                        print(f"[DEBUG] First step completed: '{first_thought[:50]}...'")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"[ERROR] First step failed: {e}")
            return f"Error in first step: {e}", "Error", []
        
        # 第二阶段：基于第一步的连续推理
        print("[DEBUG] Phase 2: Sequential reasoning based on first step")
        
        for step in range(1, min(max_steps, len(distributed_prompts))):
            try:
                step_text = distributed_prompts[step]
                print(f"[DEBUG] Starting step {step + 1}/{max_steps}")
                
                step_tokens = model.tokenizer.encode(step_text, add_special_tokens=False, return_tensors='pt')
                step_tokens = step_tokens.to(device, non_blocking=True).long()
                
                current_batch = {
                    'input_ids': torch.cat([current_prompt, step_tokens], dim=1),
                    'attention_mask': torch.ones(current_prompt.shape[0], current_prompt.shape[1] + step_tokens.shape[1], 
                                                dtype=torch.long, device=device)
                }
                
                outputs = safe_generate(
                    model,
                    input_ids=current_batch['input_ids'],
                    attention_mask=current_batch['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature * 0.8,  # 后续步骤使用稍低的temperature
                    do_sample=temperature > 0,
                    pad_token_id=model.tokenizer.pad_token_id,
                    eos_token_id=model.tokenizer.eos_token_id
                )
                
                if outputs is not None and len(outputs) > 0:
                    new_tokens = outputs[0][current_batch['input_ids'].shape[1]:]
                    if new_tokens.numel() > 0:
                        step_thought = model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                        step_thought = re.sub(r'<[^>]+>', '', step_thought).strip()
                        
                        sentences = [s.strip() for s in step_thought.split('.') if s.strip()]
                        if sentences:
                            step_thought = '. '.join(sentences[:2])  # 取前两句
                            
                            step_name = "Question Understanding" if step == 1 else "Logical Reasoning"
                            thoughts.append(f"Step {step + 1} - {step_name}: {step_thought}")
                            
                            # 更新prompt
                            step_simple = f" {step_thought}."
                            step_simple_tokens = model.tokenizer.encode(step_simple, add_special_tokens=False, return_tensors='pt')
                            step_simple_tokens = step_simple_tokens.to(device, non_blocking=True).long()
                            current_prompt = torch.cat([current_prompt, step_simple_tokens], dim=1)
                            
                            print(f"[DEBUG] Step {step + 1} completed: '{step_thought[:50]}...'")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"[ERROR] Step {step + 1} failed: {e}")
                break
        
        # 第三阶段：基于所有推理步骤生成最终答案
        print("[DEBUG] Phase 3: Final answer generation")
        final_answer = ""
        
        try:
            final_prompt_text = "\nBased on my comprehensive step-by-step analysis above, I can now provide the final answer in English, focusing only on what I can observe in the image: "
            final_tokens = model.tokenizer.encode(final_prompt_text, add_special_tokens=False, return_tensors='pt')
            final_tokens = final_tokens.to(device, non_blocking=True).long()
            
            final_batch = {
                'input_ids': torch.cat([current_prompt, final_tokens], dim=1),
                'attention_mask': torch.ones(current_prompt.shape[0], current_prompt.shape[1] + final_tokens.shape[1], 
                                            dtype=torch.long, device=device)
            }
            
            final_outputs = safe_generate(
                model,
                input_ids=final_batch['input_ids'],
                attention_mask=final_batch['attention_mask'],
                max_new_tokens=min(max_new_tokens, 30),
                temperature=0.0,  # 确定性生成最终答案
                do_sample=False,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id
            )
            
            if final_outputs is not None and len(final_outputs) > 0:
                final_new_tokens = final_outputs[0][final_batch['input_ids'].shape[1]:]
                if final_new_tokens.numel() > 0:
                    final_answer = model.tokenizer.decode(final_new_tokens, skip_special_tokens=True).strip()
                    final_answer = re.sub(r'<[^>]+>', '', final_answer).strip()
                    
                    # 提取简洁答案
                    sentences = [s.strip() for s in final_answer.split('.') if s.strip()]
                    if sentences:
                        final_answer = sentences[0]
                    
                    print(f"[DEBUG] Final answer generated: '{final_answer}'")
                    
        except Exception as e:
            print(f"[ERROR] Final answer generation failed: {e}")
        
        # 如果没有生成最终答案，从思维中提取
        if not final_answer and thoughts:
            try:
                last_thought = thoughts[-1]
                if ': ' in last_thought:
                    final_answer = last_thought.split(': ', 1)[1]
                else:
                    final_answer = "Unable to determine"
            except:
                final_answer = "Unable to determine"
        
        full_reasoning = ("\n".join(thoughts) + f"\nFinal Answer: {final_answer}") if thoughts else f"Direct Answer: {final_answer}"
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"[DEBUG] Distributed inference completed with {len(thoughts)} reasoning steps")
        return full_reasoning, clean_prediction(final_answer), thoughts
        
    except Exception as e:
        print(f"[ERROR] Distributed step-by-step inference failed: {e}")
        return f"Error: {e}", "Error", []

def distributed_step_by_step_inference_with_grounding(model, batch, max_steps=4, max_new_tokens=60, temperature=0.1):
    """分布式推理：结合VOCOT坐标定位能力的分步推理"""
    print(f"[DEBUG] Starting distributed step-by-step inference with grounding, max_steps={max_steps}")
    
    try:
        initialize_model_attributes(model)
        device = _get_device_of_model(model)
        
        if batch is None:
            return "Error: Batch is None", "Error", []
        
        batch = sanitize_batch(batch, model.tokenizer, device)
        if batch is None or 'input_ids' not in batch:
            return "Error: Batch sanitization failed", "Error", []

        thoughts = []
        current_prompt = batch['input_ids'].clone().to(device, non_blocking=True)
        if current_prompt.dtype != torch.long:
            current_prompt = current_prompt.long()
        
        # 针对VOCOT模型的分布式推理prompt模板（包含坐标定位）
        vocot_distributed_prompts = [
            "\nStep 1 - Visual Scene Analysis: Let me carefully examine this image to identify all visible objects, people, and elements. I need to understand the overall scene composition and spatial layout before proceeding.",
            "\nStep 2 - Object Localization: Now I will identify the specific locations of key objects in the image. For important objects related to the question, I should provide their approximate coordinates using <click>x, y</click> format to precisely locate them in the image.",
            "\nStep 3 - Question-Object Mapping: Based on my visual analysis and object localization, let me understand what the question is asking about and how it relates to the specific objects and their locations I identified.",
            "\nStep 4 - Spatial Reasoning & Answer: Using the spatial relationships, object locations, and visual evidence I've gathered, I can now apply logical reasoning to determine the most accurate answer."
        ]
        
        # 坐标定位质量评估函数
        def evaluate_grounding_quality(thought_text):
            """评估包含坐标定位的思维节点质量"""
            if not thought_text or len(thought_text.strip()) < 10:
                return 0.0
            
            quality_score = 0.0
            text_lower = thought_text.lower()
            
            # 基础质量评估
            length = len(thought_text.strip())
            if 20 <= length <= 150:  # 坐标定位需要更多描述
                quality_score += 0.2
            elif length > 150:
                quality_score += 0.1
            
            # 坐标定位相关关键词
            grounding_keywords = ['<click>', 'coordinate', 'location', 'position', 'located', 'at', 'coordinates']
            visual_keywords = ['see', 'image', 'picture', 'visual', 'observe', 'look', 'appear', 'visible']
            spatial_keywords = ['left', 'right', 'top', 'bottom', 'center', 'corner', 'edge', 'middle', 'above', 'below']
            reasoning_keywords = ['because', 'therefore', 'since', 'so', 'thus', 'hence', 'based on']
            
            # 检查是否包含坐标定位
            if any(kw in thought_text for kw in grounding_keywords):
                quality_score += 0.4  # 坐标定位是VOCOT的核心能力
            
            if any(kw in text_lower for kw in visual_keywords):
                quality_score += 0.15
            if any(kw in text_lower for kw in spatial_keywords):
                quality_score += 0.15
            if any(kw in text_lower for kw in reasoning_keywords):
                quality_score += 0.1
            
            return min(quality_score, 1.0)
        
        # 第一步：场景分析（独立思考）
        print("[DEBUG] Phase 1: Visual Scene Analysis")
        first_step_prompt = vocot_distributed_prompts[0]
        
        try:
            step_tokens = model.tokenizer.encode(first_step_prompt, add_special_tokens=False, return_tensors='pt')
            step_tokens = step_tokens.to(device, non_blocking=True).long()
            
            first_batch = {
                'input_ids': torch.cat([current_prompt, step_tokens], dim=1),
                'attention_mask': torch.ones(current_prompt.shape[0], current_prompt.shape[1] + step_tokens.shape[1], 
                                            dtype=torch.long, device=device)
            }
            
            outputs = safe_generate(
                model,
                input_ids=first_batch['input_ids'],
                attention_mask=first_batch['attention_mask'],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id
            )
            
            if outputs is not None and len(outputs) > 0:
                new_tokens = outputs[0][first_batch['input_ids'].shape[1]:]
                if new_tokens.numel() > 0:
                    first_thought = model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    first_thought = re.sub(r'<[^>]+>', '', first_thought).strip()
                    
                    sentences = [s.strip() for s in first_thought.split('.') if s.strip()]
                    if sentences:
                        first_thought = '. '.join(sentences[:3])  # 取前三句
                        thoughts.append(f"Step 1 - Visual Scene Analysis: {first_thought}")
                        
                        # 更新prompt
                        step_simple = f" {first_thought}."
                        step_simple_tokens = model.tokenizer.encode(step_simple, add_special_tokens=False, return_tensors='pt')
                        step_simple_tokens = step_simple_tokens.to(device, non_blocking=True).long()
                        current_prompt = torch.cat([current_prompt, step_simple_tokens], dim=1)
                        
                        print(f"[DEBUG] Scene analysis completed: '{first_thought[:60]}...'")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"[ERROR] Scene analysis failed: {e}")
            return f"Error in scene analysis: {e}", "Error", []
        
        # 第二步：物品定位（VOCOT核心能力）
        print("[DEBUG] Phase 2: Object Localization with Coordinates")
        
        try:
            localization_prompt = vocot_distributed_prompts[1]
            step_tokens = model.tokenizer.encode(localization_prompt, add_special_tokens=False, return_tensors='pt')
            step_tokens = step_tokens.to(device, non_blocking=True).long()
            
            # 生成多个候选来提高坐标定位质量
            localization_candidates = []
            for candidate_idx in range(3):
                try:
                    current_batch = {
                        'input_ids': torch.cat([current_prompt, step_tokens], dim=1),
                        'attention_mask': torch.ones(current_prompt.shape[0], current_prompt.shape[1] + step_tokens.shape[1], 
                                                    dtype=torch.long, device=device)
                    }
                    
                    # 使用稍高的temperature来鼓励坐标生成的多样性
                    temp = temperature + candidate_idx * 0.15
                    outputs = safe_generate(
                        model,
                        input_ids=current_batch['input_ids'],
                        attention_mask=current_batch['attention_mask'],
                        max_new_tokens=max_new_tokens + 20,  # 坐标定位需要更多token
                        temperature=temp,
                        do_sample=temp > 0,
                        pad_token_id=model.tokenizer.pad_token_id,
                        eos_token_id=model.tokenizer.eos_token_id
                    )
                    
                    if outputs is not None and len(outputs) > 0:
                        new_tokens = outputs[0][current_batch['input_ids'].shape[1]:]
                        if new_tokens.numel() > 0:
                            localization_thought = model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                            # 保留坐标标记，不要过滤掉<click>标签
                            localization_thought = re.sub(r'<(?!/?click)[^>]+>', '', localization_thought).strip()
                            
                            sentences = [s.strip() for s in localization_thought.split('.') if s.strip()]
                            if sentences:
                                localization_thought = '. '.join(sentences[:3])
                                quality = evaluate_grounding_quality(localization_thought)
                                localization_candidates.append((localization_thought, quality))
                                print(f"[DEBUG] Localization candidate {candidate_idx + 1}: quality={quality:.2f}")
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"[WARNING] Localization candidate {candidate_idx + 1} failed: {e}")
                    continue
            
            # 选择最佳定位候选
            if localization_candidates:
                localization_candidates.sort(key=lambda x: x[1], reverse=True)
                best_localization, best_quality = localization_candidates[0]
                
                print(f"[DEBUG] Selected best localization: quality={best_quality:.2f}")
                
                if best_quality > 0.3:
                    thoughts.append(f"Step 2 - Object Localization: {best_localization}")
                    
                    # 更新prompt
                    step_simple = f" {best_localization}."
                    step_simple_tokens = model.tokenizer.encode(step_simple, add_special_tokens=False, return_tensors='pt')
                    step_simple_tokens = step_simple_tokens.to(device, non_blocking=True).long()
                    current_prompt = torch.cat([current_prompt, step_simple_tokens], dim=1)
                else:
                    print(f"[WARNING] Localization quality too low: {best_quality:.2f}")
            else:
                print(f"[WARNING] No valid localization candidates generated")
                
        except Exception as e:
            print(f"[ERROR] Object localization failed: {e}")
        
        # 第三和第四步：问题映射和空间推理
        for step in range(2, min(max_steps, len(vocot_distributed_prompts))):
            try:
                step_text = vocot_distributed_prompts[step]
                print(f"[DEBUG] Starting step {step + 1}/{max_steps}")
                
                step_tokens = model.tokenizer.encode(step_text, add_special_tokens=False, return_tensors='pt')
                step_tokens = step_tokens.to(device, non_blocking=True).long()
                
                current_batch = {
                    'input_ids': torch.cat([current_prompt, step_tokens], dim=1),
                    'attention_mask': torch.ones(current_prompt.shape[0], current_prompt.shape[1] + step_tokens.shape[1], 
                                                dtype=torch.long, device=device)
                }
                
                outputs = safe_generate(
                    model,
                    input_ids=current_batch['input_ids'],
                    attention_mask=current_batch['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature * 0.7,  # 后续步骤使用更低temperature
                    do_sample=temperature > 0,
                    pad_token_id=model.tokenizer.pad_token_id,
                    eos_token_id=model.tokenizer.eos_token_id
                )
                
                if outputs is not None and len(outputs) > 0:
                    new_tokens = outputs[0][current_batch['input_ids'].shape[1]:]
                    if new_tokens.numel() > 0:
                        step_thought = model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                        step_thought = re.sub(r'<(?!/?click)[^>]+>', '', step_thought).strip()
                        
                        sentences = [s.strip() for s in step_thought.split('.') if s.strip()]
                        if sentences:
                            step_thought = '. '.join(sentences[:2])
                            
                            step_name = "Question-Object Mapping" if step == 2 else "Spatial Reasoning"
                            thoughts.append(f"Step {step + 1} - {step_name}: {step_thought}")
                            
                            # 更新prompt
                            step_simple = f" {step_thought}."
                            step_simple_tokens = model.tokenizer.encode(step_simple, add_special_tokens=False, return_tensors='pt')
                            step_simple_tokens = step_simple_tokens.to(device, non_blocking=True).long()
                            current_prompt = torch.cat([current_prompt, step_simple_tokens], dim=1)
                            
                            print(f"[DEBUG] Step {step + 1} completed: '{step_thought[:50]}...'")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"[ERROR] Step {step + 1} failed: {e}")
                break
        
        # 最终答案生成（结合坐标信息）
        print("[DEBUG] Phase 3: Final Answer Generation with Grounding")
        final_answer = ""
        
        try:
            final_prompt_text = "\nBased on my comprehensive visual analysis, object localization with coordinates, and spatial reasoning above, the final answer is: "
            final_tokens = model.tokenizer.encode(final_prompt_text, add_special_tokens=False, return_tensors='pt')
            final_tokens = final_tokens.to(device, non_blocking=True).long()
            
            final_batch = {
                'input_ids': torch.cat([current_prompt, final_tokens], dim=1),
                'attention_mask': torch.ones(current_prompt.shape[0], current_prompt.shape[1] + final_tokens.shape[1], 
                                            dtype=torch.long, device=device)
            }
            
            final_outputs = safe_generate(
                model,
                input_ids=final_batch['input_ids'],
                attention_mask=final_batch['attention_mask'],
                max_new_tokens=min(max_new_tokens, 40),
                temperature=0.0,  # 确定性生成最终答案
                do_sample=False,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id
            )
            
            if final_outputs is not None and len(final_outputs) > 0:
                final_new_tokens = final_outputs[0][final_batch['input_ids'].shape[1]:]
                if final_new_tokens.numel() > 0:
                    final_answer = model.tokenizer.decode(final_new_tokens, skip_special_tokens=True).strip()
                    final_answer = re.sub(r'<(?!/?click)[^>]+>', '', final_answer).strip()
                    
                    # 提取简洁答案
                    sentences = [s.strip() for s in final_answer.split('.') if s.strip()]
                    if sentences:
                        final_answer = sentences[0]
                    
                    print(f"[DEBUG] Final answer with grounding: '{final_answer}'")
                    
        except Exception as e:
            print(f"[ERROR] Final answer generation failed: {e}")
        
        # 如果没有生成最终答案，从思维中提取
        if not final_answer and thoughts:
            try:
                last_thought = thoughts[-1]
                if ': ' in last_thought:
                    final_answer = last_thought.split(': ', 1)[1]
                else:
                    final_answer = "Unable to determine"
            except:
                final_answer = "Unable to determine"
        
        full_reasoning = ("\n".join(thoughts) + f"\nFinal Answer: {final_answer}") if thoughts else f"Direct Answer: {final_answer}"
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"[DEBUG] VOCOT distributed inference completed with {len(thoughts)} reasoning steps")
        return full_reasoning, clean_prediction(final_answer), thoughts
        
    except Exception as e:
        print(f"[ERROR] VOCOT distributed step-by-step inference failed: {e}")
        return f"Error: {e}", "Error", []

# -------------------- 清理与评估 --------------------
def clean_prediction(prediction):
    if prediction is None:
        return ""
    prediction = str(prediction).strip()
    # 保留坐标信息，只移除其他HTML标签
    prediction = re.sub(r'<(?!/?(?:click|coor))[^>]*>', '', prediction)

    prefixes_to_remove = [
        'Answer:', 'The answer is:', 'A:', 'Answer is:',
        'The answer is ', 'Answer: ', 'Final answer:',
        'Final answer is:', 'Response:', 'Output:', 'Final Answer:',
        'Based on my analysis, the answer is:'
    ]
    for prefix in prefixes_to_remove:
        if prediction.lower().startswith(prefix.lower()):
            prediction = prediction[len(prefix):].strip()

    if '\n' in prediction:
        lines = [line.strip() for line in prediction.split('\n') if line.strip()]
        if lines:
            prediction = lines[-1]

    prediction = prediction.rstrip('.!?;:,').strip()
    words = prediction.split()
    if len(prediction.split()) > 10:  # 如果答案超过10个词才截断
        sentences = [s.strip() for s in prediction.split('.') if s.strip()]
        if sentences:
            prediction = sentences[0]  # 取第一句话
    
    # 最终清理，但保持原始大小写
    prediction = prediction.rstrip(r'.!?;:,"\'').strip()
    return prediction

def evaluate_gqa_with_fot(model, preprocessor, gqa_data, use_fot=True, num_thoughts=3, use_step_by_step=True, 
                         # 新增FOT参数
                         use_monte_carlo=False, num_iterations=10, ucb_constant=1.4, 
                         max_depth=5, early_stop_threshold=0.9, 
                         # 添加坐标标注参数
                         enable_coordinates=False, coordinate_confidence_threshold=0.8,
                         **kwargs):
    results = []
    correct = 0
    total = 0

    # 检查CUDA调试环境变量
    if torch.cuda.is_available():
        cuda_launch_blocking = os.environ.get('CUDA_LAUNCH_BLOCKING', '0')
        if cuda_launch_blocking != '1':
            print("Warning: CUDA_LAUNCH_BLOCKING is not set to 1. For better error debugging, consider setting:")
            print("export CUDA_LAUNCH_BLOCKING=1")
        
        # 清理CUDA缓存
        torch.cuda.empty_cache()
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    initialize_model_attributes(model)

    # 检查FOT支持类型
    has_fot_support = hasattr(model, 'fot_filter') and hasattr(model, 'condition_completion_with_fot')
    has_monte_carlo_support = hasattr(model, 'monte_carlo_forest') and hasattr(model, 'run_monte_carlo_inference')
    
    if use_fot and use_monte_carlo and not has_monte_carlo_support:
        print("Warning: Model does not support Monte Carlo FOT, falling back to standard FOT")
        use_monte_carlo = False
    elif use_fot and not has_fot_support and not has_monte_carlo_support:
        print("Warning: Model does not support FOT, falling back to standard inference")
        use_fot = False

    if isinstance(gqa_data, dict):
        questions_items = list(gqa_data.items())
    elif isinstance(gqa_data, list):
        questions_items = [(item.get('questionId', str(i)), item) for i, item in enumerate(gqa_data)]
    else:
        raise ValueError("gqa_data must be either a dict or list")

    inference_method = "Monte Carlo FOT" if (use_fot and use_monte_carlo) else ("Step-by-step FOT" if use_step_by_step else "Standard FOT")
    print(f"Starting {inference_method} evaluation on {len(questions_items)} questions...")

    data_collator = SFT_DataCollator(tokenizer=preprocessor.tokenizer, sd_tokenizer=None)
    device = _get_device_of_model(model)

    for question_id, item in tqdm(questions_items, desc=f"Evaluating with {inference_method}"):
        try:
            if isinstance(item, dict) and 'question' in item:
                question = item['question']
                answer = item.get('answer', '')
                image_id = item.get('imageId', '')
            else:
                question = item.get('question', '')
                answer = item.get('answer', '')
                image_id = item.get('imageId', item.get('image', ''))

            if not question:
                continue

            # 找图
            image_path = None
            image_dir = os.environ.get('GQA_IMAGE_DIR', '')
            candidates = [
                os.path.join(image_dir, f"{image_id}.jpg") if image_dir else None,
                image_id,
                f"{image_id}.jpg",
                os.path.join("images", f"{image_id}.jpg"),
                os.path.join("val_images", f"{image_id}.jpg"),
                os.path.join("GQA", "images", f"{image_id}.jpg"),
            ]
            candidates = [p for p in candidates if p is not None]
            for p in candidates:
                if os.path.exists(p):
                    image_path = p
                    break
            if image_path is None:
                print(f"Warning: Image not found for {image_id}")
                continue

            # 读图
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

            # 构造 query
            if use_step_by_step:
                step_instruction = "Please analyze this image carefully and answer the question. Think step by step about what you see."
                query = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + step_instruction + '\n' + question
            else:
                query = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + question + '\n' + GQA_ACTIVATION
           
           

            conv = [{'from': 'human', 'value': query}]
            item_dict = {'input_images': [image], 'conversation': conv}

            # 预处理
            try:
                input_item = preprocessor(item_dict)
                if input_item is None:
                    continue
            except Exception as e:
                print(f"Preprocessing error for question {question_id}: {e}")
                continue

            # 预验证input_item，避免在data_collator中出错
            try:
                # 检查input_item的基本结构
                if not isinstance(input_item, dict):
                    print(f"Warning: input_item is not a dict for question {question_id}")
                    continue
                
                # 检查input_ids是否存在且合理
                if 'input_ids' in input_item:
                    input_ids = input_item['input_ids']
                    if isinstance(input_ids, torch.Tensor):
                        # 限制序列长度
                        if input_ids.numel() > MAX_INPUT_LEN:
                            print(f"Warning: input_ids too long ({input_ids.numel()}) for question {question_id}, truncating")
                            input_item['input_ids'] = input_ids[:MAX_INPUT_LEN]
                        
                        # 检查token值范围
                        vocab_size = getattr(preprocessor.tokenizer, 'vocab_size', 50000)
                        if (input_ids >= vocab_size).any() or (input_ids < 0).any():
                            print(f"Warning: invalid tokens found for question {question_id}")
                            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
                            input_item['input_ids'] = input_ids
                
                # collate -> batch (强制batch_size=1)
                batch = data_collator([input_item])
                if batch is None or 'input_ids' not in batch:
                    continue
                
                # 进一步清理batch
                batch = sanitize_batch(batch, preprocessor.tokenizer, device)
                if batch is None:
                    print(f"Batch sanitize failed for question {question_id}")
                    continue
                
                # 最终验证：确保batch_size=1
                if 'input_ids' in batch and batch['input_ids'].shape[0] > 1:
                    print(f"Warning: batch_size > 1 for question {question_id}, taking first sample")
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor) and batch[key].shape[0] > 1:
                            batch[key] = batch[key][:1]
            except RuntimeError as e:
                if "device-side assert" in str(e).lower():
                    print(f"CUDA assert in batch creation for question {question_id}: {e}")
                    print("Skipping this question and continuing...")
                    # 清理CUDA缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"Runtime error in batch creation for question {question_id}: {e}")
                    continue
            except Exception as e:
                print(f"Batch creation error for question {question_id}: {e}")
                continue

            prediction = None
            full_reasoning = None
            thoughts = []
            monte_carlo_info = None

            max_new_tokens = min(kwargs.get('max_new_tokens', DEFAULT_MAX_NEW_TOKENS), 512)
            temperature = kwargs.get('temperature', DEFAULT_TEMPERATURE)

            # 优先使用蒙特卡洛FOT推理
            if use_fot and use_monte_carlo and has_monte_carlo_support:
                try:
                    print(f"Running Monte Carlo FOT inference for question {question_id}...")
                    batch = sanitize_batch(batch, preprocessor.tokenizer, device)
                    _log_batch(batch, "Monte Carlo FOT path - batch before inference")
                    
                    # 调用蒙特卡洛森林推理
                    mc_result = model.run_monte_carlo_inference(
                        batch,
                        question=question,
                        num_iterations=num_iterations,
                        ucb_constant=ucb_constant,
                        max_depth=max_depth,
                        early_stop_threshold=early_stop_threshold,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature
                    )
                    
                    if mc_result and 'best_answer' in mc_result:
                        prediction = mc_result['best_answer']
                        monte_carlo_info = {
                            'iterations_run': mc_result.get('iterations_run', 0),
                            'best_reward': mc_result.get('best_reward', 0.0),
                            'tree_size': mc_result.get('tree_size', 0),
                            'exploration_path': mc_result.get('exploration_path', []),
                            'thought_quality_scores': mc_result.get('thought_quality_scores', [])
                        }
                        print(f"Monte Carlo FOT completed: {mc_result.get('iterations_run', 0)} iterations, reward: {mc_result.get('best_reward', 0.0):.3f}")
                    else:
                        print(f"Monte Carlo FOT returned empty result for question {question_id}")
                        
                except RuntimeError as e:
                    print(f"Monte Carlo FOT runtime error for {question_id}: {e}")
                    if _is_cuda_assert(e):
                        print("Monte Carlo FOT CUDA assert, falling back to standard FOT")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Monte Carlo FOT inference failed for question {question_id}: {e}")

            # 如果蒙特卡洛FOT没有成功，继续其他推理方法
            if not prediction and use_step_by_step:
                try:
                    # 在调用step_by_step_inference前进行最后的安全检查
                    if 'input_ids' in batch:
                        input_ids = batch['input_ids']
                        seq_len = input_ids.shape[1] if len(input_ids.shape) > 1 else input_ids.shape[0]
                        
                        # 对于分步推理，使用更保守的长度限制
                        conservative_max_len = min(MAX_INPUT_LEN // 2, 2048)  # 更保守的长度限制
                        if seq_len > conservative_max_len:
                            print(f"Warning: sequence too long for step-by-step ({seq_len}), truncating to {conservative_max_len}")
                            batch['input_ids'] = input_ids[:, :conservative_max_len]
                            if 'attention_mask' in batch:
                                batch['attention_mask'] = batch['attention_mask'][:, :conservative_max_len]
                    
                    # 使用更保守的参数
                    safe_max_steps = kwargs.get('max_steps', 1)  # 使用传入的参数
                    safe_max_new_tokens = min(kwargs.get('step_max_tokens', 50), 50)  # 限制生成长度
                    
                    # 选择推理方法：分布式或标准
                    use_distributed = kwargs.get('use_distributed_inference', False)
                    if use_distributed:
                        detailed_result, prediction, thoughts = distributed_step_by_step_inference(
                            model, batch,
                            max_steps=safe_max_steps,
                            max_new_tokens=safe_max_new_tokens,
                            temperature=temperature
                        )
                    else:
                        detailed_result, prediction, thoughts = step_by_step_inference(
                            model, batch,
                            max_steps=safe_max_steps,
                            max_new_tokens=safe_max_new_tokens,
                            temperature=temperature
                        )
                    
                    # 提取详细信息
                    if isinstance(detailed_result, dict):
                        full_reasoning = detailed_result.get('full_reasoning', '')
                        final_answer = detailed_result.get('final_answer', prediction)
                    else:
                        full_reasoning = detailed_result
                        final_answer = prediction
                except RuntimeError as e:
                    print(f"Critical error in step_by_step_inference for question {question_id}: {e}")
                    if _is_cuda_assert(e):
                        print("CUDA assert in step_by_step_inference, clearing cache and skipping")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Step-by-step inference failed for question {question_id}: {e}")

            # 如果还没有预测结果，使用标准FOT或普通推理
            if not prediction:
                if use_fot and has_fot_support:
                    try:
                        # 标准FOT 路径
                        batch = sanitize_batch(batch, preprocessor.tokenizer, device)
                        _log_batch(batch, "Standard FOT path - batch before inference")
                        txt_res, out_imgs, txt_ids = model.condition_completion_with_fot(
                            batch,
                            avoid_image_gen=True,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            num_thoughts=num_thoughts
                        )
                        prediction = txt_res[0] if isinstance(txt_res, list) else txt_res
                    except RuntimeError as e:
                        print(f"Standard FOT inference runtime error for {question_id}: {e}")
                        if _is_cuda_assert(e):
                            print("Standard FOT path CUDA assert -> CPU fallback")
                            cpu = torch.device('cpu')
                            _move_model(model, cpu)
                            batch = _to_device(batch, cpu)
                            try:
                                txt_res, out_imgs, txt_ids = model.condition_completion_with_fot(
                                    batch,
                                    avoid_image_gen=True,
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature,
                                    num_thoughts=num_thoughts
                                )
                                prediction = txt_res[0] if isinstance(txt_res, list) else txt_res
                            except Exception as e2:
                                print(f"CPU FOT inference also failed: {e2}")
                    except Exception as e:
                        print(f"Standard FOT inference failed for question {question_id}: {e}")

                # 最后的标准推理fallback
                if not prediction:
                    try:
                        batch = sanitize_batch(batch, preprocessor.tokenizer, device)
                        _log_batch(batch, "Standard path - batch before inference")
                        txt_res, out_imgs, txt_ids = model.condition_completion(
                            batch,
                            avoid_image_gen=True,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature
                        )
                        prediction = txt_res[0] if isinstance(txt_res, list) else txt_res
                    except RuntimeError as e:
                        print(f"Standard inference runtime error for {question_id}: {e}")
                        if _is_cuda_assert(e):
                            print("Standard path CUDA assert -> CPU fallback")
                            cpu = torch.device('cpu')
                            _move_model(model, cpu)
                            batch = _to_device(batch, cpu)
                            try:
                                txt_res, out_imgs, txt_ids = model.condition_completion(
                                    batch,
                                    avoid_image_gen=True,
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature
                                )
                                prediction = txt_res[0] if isinstance(txt_res, list) else txt_res
                            except Exception as e2:
                                print(f"CPU standard inference also failed: {e2}")
                    except Exception as e:
                        print(f"Standard inference failed: {e}")

            # 在获得prediction后，如果启用坐标标注，则调用坐标生成函数
            coordinate_answer = None  # 添加这行来跟踪坐标答案
            if prediction and enable_coordinates:
                try:
                    # 生成带坐标的答案 - 传递完整的batch而不是query
                    coordinate_answer = generate_answer_with_coordinates(
                        model=model,
                        batch=batch,  # 传递包含图像信息的batch
                        thoughts=thoughts if thoughts else [],
                        device=device,
                        max_new_tokens=80
                    )
                    
                    # 如果坐标答案生成成功，替换原答案
                    if coordinate_answer and coordinate_answer != prediction:
                        print(f"[DEBUG] Coordinate answer generated: {coordinate_answer}")
                        prediction = coordinate_answer
                        
                except Exception as e:
                    print(f"Error generating coordinate answer for question {question_id}: {e}")
                    # 如果坐标生成失败，继续使用原答案
            
            cleaned_prediction = clean_prediction(prediction)
            cleaned_answer = (answer or '').lower().strip()
            is_correct = cleaned_prediction == cleaned_answer
            if is_correct:
                correct += 1
                
            # 构建结果记录
            result_item = {
                'questionId': question_id,
                'question': question,
                'answer': answer,
                'prediction': cleaned_prediction,
                'raw_prediction': str(prediction) if prediction else "",
                'full_reasoning': full_reasoning if use_step_by_step else None,
                'thoughts': thoughts if use_step_by_step else None,
                'correct': is_correct,
                'inference_method': inference_method,
                # 添加详细的推理信息
                'final_answer': final_answer if 'final_answer' in locals() else cleaned_prediction,
                # 使用坐标答案作为decoded_prediction，如果没有坐标答案则使用原prediction
                'decoded_prediction': str(coordinate_answer) if coordinate_answer else str(prediction) if prediction else "",
                'step_by_step_reasoning': {
                    'enabled': use_step_by_step,
                    'reasoning_steps': thoughts if thoughts else [],
                    'final_reasoning': full_reasoning if full_reasoning else ""
                },
                'generation_details': {
                    'temperature': temperature,
                    'max_new_tokens': max_new_tokens,
                    'inference_mode': inference_method
                }
            }
            
            # 添加蒙特卡洛相关信息
            if monte_carlo_info:
                result_item['monte_carlo_info'] = monte_carlo_info
                result_item['monte_carlo_details'] = {
                    'iterations_run': monte_carlo_info.get('iterations_run', 0),
                    'best_reward': monte_carlo_info.get('best_reward', 0.0),
                    'tree_size': monte_carlo_info.get('tree_size', 0),
                    'exploration_path': monte_carlo_info.get('exploration_path', []),
                    'thought_quality_scores': monte_carlo_info.get('thought_quality_scores', [])
                }
                
            results.append(result_item)
            total += 1

            if total % 50 == 0:
                acc = correct / total * 100
                print(f"Processed {total} questions, current accuracy: {acc:.2f}%")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            # 每10个问题清理一次CUDA缓存，防止内存累积
            if total % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing question {question_id}: {e}")
            continue

    accuracy = correct / total if total > 0 else 0.0
    summary = {
        'total_questions': total,
        'correct_answers': correct,
        'accuracy': accuracy,
        'fot_enabled': use_fot,
        'monte_carlo_enabled': use_monte_carlo,
        'step_by_step_enabled': use_step_by_step,
        'num_thoughts': num_thoughts if use_fot else 1,
        'monte_carlo_params': {
            'num_iterations': num_iterations,
            'ucb_constant': ucb_constant,
            'max_depth': max_depth,
            'early_stop_threshold': early_stop_threshold
        } if use_monte_carlo else None,
        'has_fot_support': has_fot_support,
        'has_monte_carlo_support': has_monte_carlo_support,
        'inference_method': inference_method,
        'results': results
    }

    print(f"\n=== {inference_method} Evaluation Completed ===")
    print(f"Total questions processed: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"FOT enabled: {use_fot}")
    print(f"Monte Carlo enabled: {use_monte_carlo}")
    print(f"Step-by-step reasoning enabled: {use_step_by_step}")
    
    if use_monte_carlo and monte_carlo_info:
        # 统计蒙特卡洛相关信息
        mc_results = [r.get('monte_carlo_info') for r in results if r.get('monte_carlo_info')]
        if mc_results:
            avg_iterations = sum(r['iterations_run'] for r in mc_results) / len(mc_results)
            avg_reward = sum(r['best_reward'] for r in mc_results) / len(mc_results)
            avg_tree_size = sum(r['tree_size'] for r in mc_results) / len(mc_results)
            print(f"Monte Carlo Statistics:")
            print(f"  Average iterations: {avg_iterations:.1f}")
            print(f"  Average best reward: {avg_reward:.3f}")
            print(f"  Average tree size: {avg_tree_size:.1f}")

    return summary

def multi_aspect_inference_with_grounding(model, batch, max_steps=1, max_new_tokens=60, temperature=0.1):
    """
    实现多角度分析的FOT推理，每个分析大类内包含3个不同子维度的候选思维
    """
    try:
        initialize_model_attributes(model)
        device = _get_device_of_model(model)
        batch = sanitize_batch(batch, model.tokenizer, device)
        
        # 定义分析大类及其子维度
        analysis_categories = {
            "visual_analysis": {
                "objects_entities": "Focus on identifying specific objects, people, animals, and their detailed characteristics. Use <click>x, y</click> to mark each important entity.",
                "background_environment": "Analyze the background, setting, environment, and spatial context. Use <click>x, y</click> to mark environmental features.",
                "actions_interactions": "Examine actions, movements, interactions, and dynamic relationships between elements. Use <click>x, y</click> to highlight action areas."
            },
            "question_analysis": {
                "question_meaning": "Analyze what the question is specifically asking about, including the main subject and required information.",
                "contextual_relationships": "Examine the relationships, connections, and contextual clues that are relevant to answering the question.",
                "answer_format": "Determine the expected answer type (yes/no, count, description, comparison, etc.) and structure the response accordingly."
            },
            "logical_reasoning": {
                "direct_observation": "Provide reasoning based on direct visual evidence that can be clearly observed in the image.",
                "inferential_reasoning": "Apply logical inference and common knowledge to derive conclusions from visual clues.",
                "comparative_analysis": "Compare different elements, consider alternatives, and evaluate the most likely answer based on evidence."
            }
        }
        
        all_thoughts = []
        all_scores = []
        
        # 为每个分析大类生成候选思维
        for category_name, sub_dimensions in analysis_categories.items():
            category_thoughts = []
            
            # 为每个子维度生成一个候选思维
            for dimension_name, dimension_prompt in sub_dimensions.items():
                prompt = f"""
{GQA_ACTIVATION}

Analysis Category: {category_name.replace('_', ' ').title()}
Specific Focus: {dimension_name.replace('_', ' ').title()}

{dimension_prompt}

Question: {batch.get('question', '')}

Provide detailed analysis from this specific perspective:
"""
                
                try:
                    thought_batch = {
                        'input_ids': batch['input_ids'],
                        'attention_mask': batch['attention_mask'],
                        'images': batch.get('images'),
                        'prompt': prompt
                    }
                    
                    result = safe_generate(
                        model,
                        **thought_batch,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=model.tokenizer.eos_token_id
                    )
                    
                    if result and 'generated_text' in result:
                        thought_text = result['generated_text'].strip()
                        if thought_text:
                            score = evaluate_aspect_quality(thought_text, dimension_name)
                            thought_data = {
                                'text': thought_text,
                                'category': category_name,
                                'dimension': dimension_name,
                                'score': score
                            }
                            category_thoughts.append(thought_data)
                            all_thoughts.append(thought_data)
                            all_scores.append(score)
                            
                except Exception as e:
                    print(f"Error generating thought for {category_name}-{dimension_name}: {e}")
                    continue
            
            # 确保每个类别至少有一个思维
            if not category_thoughts:
                print(f"Warning: No valid thoughts generated for category {category_name}")
        
        # 选择最佳思维组合
        if not all_thoughts:
            return {
                'reasoning': "No valid thoughts generated",
                'prediction': "Unable to analyze",
                'thoughts': [],
                'confidence': 0.0
            }
        
        # 按分数排序并选择最佳思维
        all_thoughts.sort(key=lambda x: x['score'], reverse=True)
        best_thoughts = all_thoughts[:6]  # 选择前6个最佳思维（每个大类最多2个）
        
        # 构建综合推理
        combined_reasoning = "Multi-Dimensional Analysis:\n\n"
        
        # 按类别组织思维
        thoughts_by_category = {}
        for thought in best_thoughts:
            category = thought['category']
            if category not in thoughts_by_category:
                thoughts_by_category[category] = []
            thoughts_by_category[category].append(thought)
        
        for category_name, category_thoughts in thoughts_by_category.items():
            combined_reasoning += f"**{category_name.replace('_', ' ').title()}:**\n"
            for i, thought in enumerate(category_thoughts, 1):
                dimension_name = thought['dimension'].replace('_', ' ').title()
                combined_reasoning += f"{i}. {dimension_name}: {thought['text']}\n\n"
        
        # 生成最终答案
        final_prompt = f"""
{GQA_ACTIVATION}

Based on the comprehensive multi-dimensional analysis above:
{combined_reasoning}

Question: {question_text if 'question_text' in locals() else 'the given question'}

IMPORTANT: Provide a clear, direct answer in English only. Focus ONLY on what is visible in the image. Avoid abstract concepts like dreams, imagination, fantasy, or emotions.

Answer:"""
        
        try:
            final_batch = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
                'images': batch.get('images'),
                'prompt': final_prompt
            }
            
            final_result = safe_generate(
                model,
                **final_batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature * 0.8,  # 降低温度以获得更稳定的最终答案
                do_sample=False,
                pad_token_id=model.tokenizer.eos_token_id
            )
            
            if final_result and 'generated_text' in final_result:
                final_answer = final_result['generated_text'].strip()
            else:
                final_answer = extract_answer_from_thoughts(best_thoughts, batch.get('question', ''))
                
        except Exception as e:
            print(f"Error generating final answer: {e}")
            final_answer = extract_answer_from_thoughts(best_thoughts, batch.get('question', ''))
        
        # 清理预测结果
        cleaned_prediction = clean_prediction_enhanced(final_answer, batch.get('question', ''))
        
        # 计算置信度
        confidence = np.mean(all_scores) if all_scores else 0.0
        
        return {
            'reasoning': combined_reasoning + f"\n**Final Answer:** {final_answer}",
            'prediction': cleaned_prediction,
            'thoughts': all_thoughts,
            'best_thoughts': best_thoughts,
            'confidence': confidence,
            'category_distribution': {cat: len(thoughts) for cat, thoughts in thoughts_by_category.items()}
        }
        
    except Exception as e:
        print(f"Error in multi_aspect_inference_with_grounding: {e}")
        return {
            'reasoning': f"Error during multi-dimensional analysis: {str(e)}",
            'prediction': "Error",
            'thoughts': [],
            'confidence': 0.0
        }

def evaluate_aspect_quality(thought_text, aspect_name):
    """
    评估特定角度思维的质量
    """
    if not thought_text or len(thought_text.strip()) < 10:
        return 0.1
    
    score = 0.5  # 基础分数
    text_lower = thought_text.lower()
    
    # 长度奖励（适中长度最佳）
    length = len(thought_text)
    if 50 <= length <= 200:
        score += 0.2
    elif 30 <= length <= 300:
        score += 0.1
    
    # 坐标定位奖励
    if '<click>' in thought_text and '</click>' in thought_text:
        score += 0.3
    
    # 角度特定关键词奖励
    aspect_keywords = {
        "人物物体": ["person", "people", "man", "woman", "child", "object", "item", "thing", "wearing", "holding"],
        "环境场景": ["background", "setting", "environment", "scene", "location", "place", "room", "outdoor", "indoor"],
        "交互关系": ["interaction", "relationship", "between", "with", "together", "near", "next to", "behind", "in front"],
        "问题焦点": ["question", "asking", "focus", "main", "primary", "key", "important", "specifically"],
        "上下文含义": ["context", "meaning", "implies", "suggests", "indicates", "overall", "broader", "general"],
        "答案格式": ["yes", "no", "count", "number", "describe", "explanation", "because", "answer"]
    }
    
    if aspect_name in aspect_keywords:
        keyword_count = sum(1 for keyword in aspect_keywords[aspect_name] if keyword in text_lower)
        score += min(keyword_count * 0.1, 0.3)
    
    # 视觉描述奖励
    visual_keywords = ["see", "visible", "shown", "appears", "looks", "image", "picture", "color", "shape"]
    visual_count = sum(1 for keyword in visual_keywords if keyword in text_lower)
    score += min(visual_count * 0.05, 0.2)
    
    return min(score, 1.0)

def extract_answer_from_thoughts(thoughts, question):
    """
    从思维中提取答案
    """
    if not thoughts:
        return "Unable to determine"
    
    # 处理不同格式的thoughts
    if isinstance(thoughts, list) and thoughts:
        if isinstance(thoughts[0], dict):
            # 如果thoughts是字典列表，优先从问题分析阶段的思维中提取
            question_thoughts = [t for t in thoughts if t.get('stage') == 'question_analysis']
            if question_thoughts:
                best_thought = question_thoughts[0].get('text', '')
            else:
                best_thought = thoughts[0].get('text', str(thoughts[0]))
        else:
            # 如果thoughts是字符串列表
            best_thought = thoughts[0] if thoughts else ""
    else:
        best_thought = str(thoughts)
    
    # 尝试提取明确的答案
    answer_patterns = [
        r'answer[:\s]+([^.\n]+)',
        r'the answer is[:\s]+([^.\n]+)',
        r'therefore[,\s]+([^.\n]+)',
        r'so[,\s]+([^.\n]+)',
        r'thus[,\s]+([^.\n]+)'
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, best_thought, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # 如果是yes/no问题，优先查找yes/no
    if any(word in question.lower() for word in ['is', 'are', 'does', 'do', 'can', 'will', 'would']):
        if 'yes' in best_thought.lower():
            return 'yes'
        elif 'no' in best_thought.lower():
            return 'no'
    
    # 返回思维的最后一句话
    sentences = best_thought.split('.')
    return sentences[-1].strip() if sentences else best_thought

def generate_answer_with_coordinates(model, batch, thoughts, device, max_new_tokens=80):
    """
    生成包含坐标标注的答案
    将检测到的物体/人物与其坐标信息结合，生成格式化的答案
    """
    try:
        # 首先生成基础答案
        base_answer = extract_answer_from_thoughts(thoughts, "")
        if not base_answer:
            base_answer = "Unable to determine answer"
        
        print(f"[DEBUG] Base answer extracted: {base_answer}")
        
        # 构建包含约束条件的坐标生成提示
        coordinate_instruction = f"""Answer: {base_answer}

IMPORTANT CONSTRAINTS:
1. RESPOND ONLY IN ENGLISH - Do not use any other language
2. FOCUS ONLY ON VISIBLE ELEMENTS - Only describe what you can actually see in the image
3. FORBIDDEN WORDS - Never use words like: dreams, imagination, fantasy, metaphor, symbolism, or any abstract concepts
4. STICK TO FACTS - Only mention concrete, visible objects, people, actions, colors, and spatial relationships

Now add bounding box coordinates for objects mentioned in the answer. Use format: object <coor>x1,y1,x2,y2</coor>
Coordinates are normalized 0.0-1.0, where (x1,y1) is top-left, (x2,y2) is bottom-right.

Final answer with coordinates (English only, visible elements only):"""
        
        # 获取原始input_ids并添加坐标指令
        original_input_ids = batch['input_ids']
        
        # 将坐标指令编码
        coord_tokens = model.tokenizer(
            "\n" + coordinate_instruction,
            return_tensors="pt",
            add_special_tokens=False
        )['input_ids'].to(device)
        
        # 拼接原始输入和坐标指令
        new_input_ids = torch.cat([original_input_ids, coord_tokens], dim=1)
        
        # 更新batch
        modified_batch = batch.copy()
        modified_batch['input_ids'] = new_input_ids
        
        # 更新attention_mask
        if 'attention_mask' in batch:
            coord_attention = torch.ones_like(coord_tokens)
            modified_batch['attention_mask'] = torch.cat([
                batch['attention_mask'], 
                coord_attention
            ], dim=1)
        
        # 确保batch在正确设备上
        modified_batch = sanitize_batch(modified_batch, model.tokenizer, device)
        if modified_batch is None:
            return base_answer
            
        # 使用模型的标准生成方法（包含图像处理）
        with torch.no_grad():
            try:
                # 使用模型的condition_completion方法，这样可以处理图像
                txt_res, _, _ = model.condition_completion(
                    modified_batch,
                    avoid_image_gen=True,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1
                )
                generated_text = txt_res[0] if isinstance(txt_res, list) else txt_res
                print(f"[DEBUG] Raw generated text: {generated_text}")
            except Exception as e:
                print(f"Error in condition_completion: {e}")
                # 回退到safe_generate
                outputs = safe_generate(
                    model,
                    input_ids=modified_batch['input_ids'],
                    attention_mask=modified_batch['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=False
                )
                if outputs is not None:
                    generated_text = model.tokenizer.decode(
                        outputs[0][modified_batch['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    ).strip()
                    print(f"[DEBUG] Raw generated text from safe_generate: {generated_text}")
                else:
                    return base_answer
        
        # 清理和格式化输出
        coordinate_answer = clean_coordinate_answer(generated_text)
        print(f"[DEBUG] Cleaned coordinate answer: {coordinate_answer}")
        
        # 验证坐标格式是否正确，如果没有坐标或格式错误，返回基础答案
        if not coordinate_answer or coordinate_answer.strip() == generated_text.strip():
            print(f"[DEBUG] No valid coordinates found, returning base answer: {base_answer}")
            return base_answer
        
        # 检查是否包含有效的坐标标签
        if '<coor>' not in coordinate_answer:
            print(f"[DEBUG] No coordinate tags found, returning base answer: {base_answer}")
            return base_answer
            
        print(f"[DEBUG] Final coordinate answer: {coordinate_answer}")
        return coordinate_answer
        
    except Exception as e:
        print(f"Error generating coordinate answer: {e}")
        # 如果生成失败，返回基础答案
        if thoughts:
            return extract_answer_from_thoughts(thoughts, "")
        return "Error in coordinate generation"

def clean_coordinate_answer(text):
    """
    清理坐标答案，移除多余的指令和格式化文本，过滤禁用词汇
    """
    if not text:
        return text
    
    # 定义禁用词汇列表
    forbidden_words = [
        'dreams', 'dream', 'dreaming', 'imagination', 'imagine', 'fantasy', 
        'metaphor', 'symbolism', 'symbolic', 'abstract', 'conceptual',
        'spiritual', 'mystical', 'magical', 'surreal', 'ethereal'
    ]
    
    # 移除常见的指令性文本
    instruction_patterns = [
        r'Answer:\s*',
        r'Final answer with coordinates.*?:',
        r'IMPORTANT CONSTRAINTS:.*?(?=\n\n|$)',
        r'\d+\.\s*[A-Z][^\n]*\n',  # 移除编号的约束条件
        r'Now add bounding box coordinates.*?\n',
        r'Coordinates are normalized.*?\n',
        r'Use format:.*?\n',
        r'Step \d+.*?:',
        r'\[DEBUG\].*?\n',
    ]
    
    cleaned = text
    for pattern in instruction_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # 检查并过滤禁用词汇
    for word in forbidden_words:
        if word.lower() in cleaned.lower():
            print(f"[WARNING] Forbidden word '{word}' detected in answer, filtering...")
            # 移除包含禁用词汇的句子
            sentences = cleaned.split('.')
            filtered_sentences = []
            for sentence in sentences:
                if word.lower() not in sentence.lower():
                    filtered_sentences.append(sentence)
                else:
                    print(f"[WARNING] Removed sentence containing '{word}': {sentence.strip()}")
            cleaned = '.'.join(filtered_sentences)
    
    # 移除多余的换行和空格
    cleaned = re.sub(r'\n+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    
    # 标准化坐标格式
    cleaned = re.sub(r'<coor>\s*([0-9.,\s]+)\s*</coor>', r'<coor>\1</coor>', cleaned)
    
    # 验证并修复坐标格式
    def fix_coordinates(match):
        coords_str = match.group(1).strip()
        try:
            coords = [float(x.strip()) for x in coords_str.split(',') if x.strip()]
            
            if len(coords) == 4:
                # 确保坐标在0-1范围内
                coords = [max(0.0, min(1.0, c)) for c in coords]
                # 确保x1 < x2, y1 < y2
                x1, y1, x2, y2 = coords
                if x1 >= x2:
                    x1, x2 = max(0.0, x2 - 0.1), min(1.0, x1 + 0.1)
                if y1 >= y2:
                    y1, y2 = max(0.0, y2 - 0.1), min(1.0, y1 + 0.1)
                return f'<coor>{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}</coor>'
        except ValueError:
            print(f"[WARNING] Invalid coordinate format: {coords_str}")
        
        return match.group(0)  # 返回原始匹配如果格式不正确
    
    cleaned = re.sub(r'<coor>([0-9.,\s]+)</coor>', fix_coordinates, cleaned)
    
    return cleaned

def validate_bounding_box_format(text):
    """
    验证文本中的坐标是否为正确的边界框格式
    """
    if not text:
        return False
    
    # 查找所有坐标标签
    coord_pattern = r'<coor>([0-9.,\s]+)</coor>'
    matches = re.findall(coord_pattern, text)
    
    if not matches:
        return True  # 如果没有坐标标签，也算有效
    
    for coord_str in matches:
        coords = coord_str.strip().split(',')
        if len(coords) != 4:
            return False
        
        try:
            x1, y1, x2, y2 = [float(c.strip()) for c in coords]
            # 验证坐标范围和逻辑
            if not (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1):
                return False
            if x1 >= x2 or y1 >= y2:  # 确保是有效的边界框
                return False
        except ValueError:
            return False
    
    return True

def clean_prediction_enhanced(prediction, question=""):
    """
    增强的预测结果清理函数，更好地处理多角度分析结果
    """
    if not prediction:
        return ""
    
    # 移除HTML标签
    prediction = re.sub(r'<[^>]+>', '', prediction)
    
    # 移除坐标标记但保留内容
    prediction = re.sub(r'<click>\d+,\s*\d+</click>', '', prediction)
    
    # 强制移除所有中文字符（包括标点符号）
    prediction = re.sub(r'[\u4e00-\u9fff\uff00-\uffef]', '', prediction)
    
    # 移除禁用词汇
    forbidden_words = ['dreams', 'dream', 'imagination', 'fantasy', 'emotion', 'feeling', 'think', 'believe', 'seem', 'appear']
    
    # 移除包含禁用词汇的句子
    sentences = prediction.split('.')
    filtered_sentences = []
    for sentence in sentences:
        sentence_clean = sentence.strip()
        if sentence_clean and not any(word.lower() in sentence_clean.lower() for word in forbidden_words):
            # 确保句子只包含英文字符、数字和基本标点
            if re.match(r'^[a-zA-Z0-9\s.,!?;:()\-\'"]+$', sentence_clean):
                filtered_sentences.append(sentence_clean)
    
    if filtered_sentences:
        prediction = '. '.join(filtered_sentences)
    else:
        # 如果所有句子都被过滤，返回安全的默认答案
        prediction = "Unable to determine from the image"
    
    # 移除常见前缀
    prefixes_to_remove = [
        "Based on the image, ",
        "Looking at the image, ",
        "In the image, ",
        "The image shows ",
        "I can see ",
        "The answer is ",
        "Answer: "
    ]
    
    for prefix in prefixes_to_remove:
        if prediction.lower().startswith(prefix.lower()):
            prediction = prediction[len(prefix):]
            break
    
    # 处理多行输出，取最后一个有意义的行
    lines = [line.strip() for line in prediction.split('\n') if line.strip()]
    if lines:
        prediction = lines[-1]
    
    # 对于yes/no问题，优先保留yes/no答案
    question_lower = question.lower()
    if any(word in question_lower for word in ['is', 'are', 'does', 'do', 'can', 'will', 'would']):
        prediction_lower = prediction.lower()
        if 'yes' in prediction_lower and 'no' not in prediction_lower:
            return 'yes'
        elif 'no' in prediction_lower and 'yes' not in prediction_lower:
            return 'no'
        elif 'yes' in prediction_lower and 'no' in prediction_lower:
            # 如果同时包含yes和no，选择最后出现的
            yes_pos = prediction_lower.rfind('yes')
            no_pos = prediction_lower.rfind('no')
            return 'yes' if yes_pos > no_pos else 'no'
    
    # 对于计数问题，提取数字
    if any(word in question_lower for word in ['how many', 'count', 'number']):
        numbers = re.findall(r'\b\d+\b', prediction)
        if numbers:
            return numbers[-1]
    
    # 清理标点符号和多余空格
    prediction = re.sub(r'[.!?]+$', '', prediction)  # 移除结尾标点
    prediction = re.sub(r'\s+', ' ', prediction)  # 合并多个空格
    prediction = prediction.strip()
    
    # 如果预测结果太长，保留前面的关键部分
    words = prediction.split()
    if len(words) > 10:
        # 对于长答案，保留前8个词
        prediction = ' '.join(words[:8])
    
    # 最终验证：确保输出只包含英文
    if not re.match(r'^[a-zA-Z0-9\s.,!?;:()\-\'"]+$', prediction):
        prediction = "Unable to determine from the image"
    
    return prediction.lower()

def evaluate_gqa_dataset_with_fot(model, preprocessor, gqa_data_path, output_path=None,
                                  use_fot=True, num_thoughts=3, confidence_threshold=0.7,
                                  coherence_threshold=0.6, max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                                  use_step_by_step=True, max_steps=1, use_distributed_inference=False,
                                  use_monte_carlo=False, num_iterations=10, ucb_constant=1.4,
                                  max_depth=5, early_stop_threshold=0.9, 
                                  enable_coordinates=False, coordinate_confidence_threshold=0.8,
                                  **kwargs):
    if hasattr(model, 'fot_filter') and use_fot:
        model.fot_filter.confidence_threshold = confidence_threshold
        model.fot_filter.coherence_threshold = coherence_threshold
        print(f"FOT filter configured: confidence={confidence_threshold}, coherence={coherence_threshold}")

    print(f"Loading GQA data from {gqa_data_path}...")
    with open(gqa_data_path, 'r', encoding='utf-8') as f:
        gqa_data = json.load(f)
    print(f"Loaded {len(gqa_data)} questions")

    results = evaluate_gqa_with_fot(
        model=model,
        preprocessor=preprocessor,
        gqa_data=gqa_data,
        use_fot=use_fot,
        num_thoughts=num_thoughts,
        use_step_by_step=use_step_by_step,
        max_steps=max_steps,
        max_new_tokens=max_new_tokens,
        use_monte_carlo=use_monte_carlo,
        num_iterations=num_iterations,
        ucb_constant=ucb_constant,
        max_depth=max_depth,
        early_stop_threshold=early_stop_threshold,
        enable_coordinates=enable_coordinates,
        coordinate_confidence_threshold=coordinate_confidence_threshold,
        **kwargs
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")

    return results

# 兼容 Python 3.8 无 contextlib.nullcontext
class nullcontext:
    def __init__(self): pass
    def __enter__(self): return None
    def __exit__(self, *exc): return False

def enhanced_fot_inference_with_coordination(model, batch, max_steps=3, max_new_tokens=60, temperature=0.1):
    """
    创新的FOT推理：候选关联 + 坐标生成 + BFS评分机制
    """
    print(f"[DEBUG] Starting Enhanced FOT with Coordination - max_steps={max_steps}")
    
    try:
        initialize_model_attributes(model)
        device = _get_device_of_model(model)
        batch = sanitize_batch(batch, model.tokenizer, device)
        
        if batch is None or 'input_ids' not in batch:
            return "Error: Invalid batch", "Error", []
        
        # FOT树结构：每个节点包含多个候选和它们的关联
        fot_tree = {
            'nodes': [],  # 所有节点
            'current_level': 0,
            'best_path': [],
            'coordinate_history': []  # 坐标生成历史
        }
        
        current_prompt = batch['input_ids'].clone().to(device, non_blocking=True)
        if current_prompt.dtype != torch.long:
            current_prompt = current_prompt.long()
        
        # 定义增强的分析维度（包含坐标生成角度）
        enhanced_dimensions = [
            {
                "name": "Visual Analysis",
                "candidates": [
                    {
                        "name": "Object Detection",
                        "prompt": "\nStep {step} - Object Detection: Identify all visible objects, people, and entities in the image. Focus on concrete, observable elements.",
                        "keywords": ['object', 'person', 'item', 'entity', 'visible', 'see'],
                        "weight": 1.0
                    },
                    {
                        "name": "Scene Context", 
                        "prompt": "\nStep {step} - Scene Context: Analyze the environment, setting, and spatial relationships in the image.",
                        "keywords": ['scene', 'environment', 'setting', 'background', 'location'],
                        "weight": 0.9
                    },
                    {
                        "name": "Visual Features",
                        "prompt": "\nStep {step} - Visual Features: Examine colors, lighting, composition, and visual characteristics.",
                        "keywords": ['color', 'light', 'bright', 'visual', 'appearance'],
                        "weight": 0.8
                    }
                ]
            },
            {
                "name": "Question Analysis", 
                "candidates": [
                    {
                        "name": "Question Focus",
                        "prompt": "\nStep {step} - Question Focus: Identify what the question is specifically asking about.",
                        "keywords": ['question', 'asking', 'focus', 'what', 'how', 'where'],
                        "weight": 1.0
                    },
                    {
                        "name": "Key Terms",
                        "prompt": "\nStep {step} - Key Terms: Analyze important keywords and concepts in the question.",
                        "keywords": ['keyword', 'term', 'important', 'concept'],
                        "weight": 0.9
                    },
                    {
                        "name": "Answer Type",
                        "prompt": "\nStep {step} - Answer Type: Determine the expected answer format and type.",
                        "keywords": ['answer', 'type', 'yes', 'no', 'count', 'describe'],
                        "weight": 0.8
                    }
                ]
            },
            {
                "name": "Coordinate Generation",  # 新增坐标生成维度
                "candidates": [
                    {
                        "name": "Global Coordinates",
                        "prompt": "\nStep {step} - Coordinate Generation: Identify coordinates (x1,y1,x2,y2) for ALL major objects in the image. Format: object_name(x1,y1,x2,y2)",
                        "keywords": ['coordinate', 'location', 'position', 'x1', 'y1', 'x2', 'y2'],
                        "weight": 1.2,  # 坐标生成权重更高
                        "is_coordinate": True
                    }
                ]
            }
        ]
        
        # BFS评分函数（更接近FOT机制）
        def calculate_bfs_score(candidate, step, dimension_name, all_candidates_in_step):
            """
            基于BFS机制的评分函数
            """
            base_score = 0.0
            text = candidate['text'].lower()
            
            # 1. 内容质量评分（40%）
            content_score = 0.0
            if len(candidate['text'].strip()) >= 10:
                content_score += 0.2
            if len(candidate['text'].split()) >= 5:
                content_score += 0.2
            
            # 关键词匹配
            keyword_matches = sum(1 for kw in candidate['keywords'] if kw in text)
            content_score += min(keyword_matches * 0.1, 0.3)
            
            # 2. 探索价值评分（30%）- BFS核心
            exploration_score = 0.0
            
            # 多样性奖励：与同步候选的差异性
            diversity_bonus = 0.0
            for other_candidate in all_candidates_in_step:
                if other_candidate != candidate:
                    # 计算文本相似度（简单的词汇重叠）
                    words1 = set(candidate['text'].lower().split())
                    words2 = set(other_candidate['text'].lower().split())
                    overlap = len(words1 & words2) / max(len(words1 | words2), 1)
                    diversity_bonus += (1 - overlap) * 0.1
            
            exploration_score = min(diversity_bonus, 0.3)
            
            # 3. 坐标生成特殊奖励（20%）
            coordinate_score = 0.0
            if candidate.get('is_coordinate', False):
                # 检查坐标格式
                coord_pattern = r'\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
                coord_matches = re.findall(coord_pattern, candidate['text'])
                coordinate_score = min(len(coord_matches) * 0.1, 0.2)
            
            # 4. 步骤适应性评分（10%）
            step_score = 0.0
            if step == 1:  # 第一步重视全面性
                if any(word in text for word in ['all', 'overall', 'general', 'entire']):
                    step_score += 0.1
            else:  # 后续步骤重视针对性
                if any(word in text for word in ['specific', 'particular', 'focus', 'relevant']):
                    step_score += 0.1
            
            # 综合评分
            total_score = (content_score * 0.4 + 
                          exploration_score * 0.3 + 
                          coordinate_score * 0.2 + 
                          step_score * 0.1) * candidate['weight']
            
            return min(total_score, 1.0)
        
        # 候选关联融合函数
        def fuse_candidates(selected_candidate, other_candidates, fusion_ratio=0.3):
            """
            将其他优秀候选的内容融合到选中的候选中
            """
            if not other_candidates:
                return selected_candidate['text']
            
            # 选择评分较高的其他候选进行融合
            fusion_candidates = [c for c in other_candidates 
                               if c != selected_candidate and c['score'] > 0.5]
            
            if not fusion_candidates:
                return selected_candidate['text']
            
            # 按评分排序，选择最佳的1-2个进行融合
            fusion_candidates.sort(key=lambda x: x['score'], reverse=True)
            top_fusion = fusion_candidates[:2]
            
            # 构建融合文本
            fused_text = selected_candidate['text']
            
            for fc in top_fusion:
                # 提取关键信息进行融合
                fc_words = fc['text'].split()
                if len(fc_words) > 3:
                    # 取前几个关键词或短语
                    key_phrase = ' '.join(fc_words[:min(5, len(fc_words))])
                    if key_phrase.lower() not in fused_text.lower():
                        fused_text += f" Additionally, {key_phrase.lower()}."
            
            return fused_text
        
        # 主要的BFS推理循环
        for step in range(max_steps):
            print(f"[DEBUG] Enhanced FOT Step {step + 1}/{max_steps}")
            
            step_candidates = []
            
            # 为每个维度生成候选
            for dim_idx, dimension in enumerate(enhanced_dimensions):
                print(f"[DEBUG] Processing dimension: {dimension['name']}")
                
                for candidate_template in dimension['candidates']:
                    try:
                        # 对于坐标候选：第一步全局坐标，后续步骤偏相关坐标
                        prompt_text = candidate_template['prompt']
                        if candidate_template.get('is_coordinate', False) and step > 0:
                            prompt_text = "\nStep {step} - Coordinate Generation: Identify coordinates (x1,y1,x2,y2) for objects RELEVANT to the question. Format: object_name(x1,y1,x2,y2)"
                        
                        step_text = prompt_text.format(step=step + 1)
                        
                        step_tokens = model.tokenizer.encode(step_text, add_special_tokens=False, return_tensors='pt')
                        if step_tokens is None or step_tokens.numel() == 0:
                            continue
                        step_tokens = step_tokens.to(device, non_blocking=True).long()
                        
                        # 构建当前批次
                        current_batch = {
                            'input_ids': torch.cat([current_prompt, step_tokens], dim=1),
                            'attention_mask': torch.ones(current_prompt.shape[0], 
                                                        current_prompt.shape[1] + step_tokens.shape[1], 
                                                        dtype=torch.long, device=device)
                        }
                        
                        # 生成候选
                        outputs = safe_generate(
                            model,
                            input_ids=current_batch['input_ids'],
                            attention_mask=current_batch['attention_mask'],
                            max_new_tokens=max_new_tokens,
                            temperature=temperature + dim_idx * 0.02,  # 轻微的温度变化
                            do_sample=(temperature + dim_idx * 0.02) > 0,
                            pad_token_id=model.tokenizer.pad_token_id,
                            eos_token_id=model.tokenizer.eos_token_id
                        )
                        
                        if outputs is not None and len(outputs) > 0:
                            new_tokens = outputs[0][current_batch['input_ids'].shape[1]:]
                            if new_tokens.numel() > 0:
                                candidate_text = model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                                candidate_text = re.sub(r'<[^>]+>', '', candidate_text).strip()
                                
                                # 处理候选文本
                                sentences = [s.strip() for s in candidate_text.split('.') if s.strip()]
                                if sentences:
                                    final_text = sentences[0]
                                    
                                    candidate = {
                                        'text': final_text,
                                        'dimension': dimension['name'],
                                        'candidate_name': candidate_template['name'],
                                        'keywords': candidate_template['keywords'],
                                        'weight': candidate_template['weight'],
                                        'is_coordinate': candidate_template.get('is_coordinate', False),
                                        'step': step + 1,
                                        'score': 0.0  # 将在后面计算
                                    }
                                    
                                    step_candidates.append(candidate)
                        
                        # 清理内存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"[WARNING] Candidate generation failed: {e}")
                        continue
            
            # 使用BFS评分机制为所有候选评分
            for candidate in step_candidates:
                candidate['score'] = calculate_bfs_score(candidate, step + 1, 
                                                       candidate['dimension'], step_candidates)
            
            # 按评分排序
            step_candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # 选择最佳候选并进行融合
            if step_candidates:
                best_candidate = step_candidates[0]
                
                # 候选关联融合
                fused_text = fuse_candidates(best_candidate, step_candidates[1:5])  # 与前4个候选融合
                
                print(f"[DEBUG] Step {step + 1} selected: {best_candidate['candidate_name']} "
                      f"(score: {best_candidate['score']:.3f})")
                print(f"[DEBUG] Original: {best_candidate['text']}")
                print(f"[DEBUG] Fused: {fused_text}")
                
                # 记录到FOT树
                node = {
                    'step': step + 1,
                    'selected_candidate': best_candidate,
                    'all_candidates': step_candidates,
                    'fused_text': fused_text,
                    'bfs_scores': [c['score'] for c in step_candidates]
                }
                fot_tree['nodes'].append(node)
                
                # 更新坐标历史
                if best_candidate.get('is_coordinate', False):
                    coord_pattern = r'\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
                    coords = re.findall(coord_pattern, fused_text)
                    if coords:
                        fot_tree['coordinate_history'].extend(coords)
                
                # 更新prompt
                step_update = f" {fused_text}."
                step_tokens = model.tokenizer.encode(step_update, add_special_tokens=False, return_tensors='pt')
                if step_tokens is not None and step_tokens.numel() > 0:
                    step_tokens = step_tokens.to(device, non_blocking=True).long()
                    current_prompt = torch.cat([current_prompt, step_tokens], dim=1)
            
            else:
                print(f"[WARNING] No valid candidates for step {step + 1}")
        
        # 生成最终答案
        print(f"[DEBUG] Generating final answer with Enhanced FOT")
        
        # 构建综合推理文本
        reasoning_parts = []
        for i, node in enumerate(fot_tree['nodes']):
            selected = node['selected_candidate']
            reasoning_parts.append(
                f"Step {i+1} ({selected['dimension']} - {selected['candidate_name']}): {node['fused_text']}"
            )
        
        # 坐标总结
        if fot_tree['coordinate_history']:
            coord_summary = f"Identified coordinates: {fot_tree['coordinate_history']}"
            reasoning_parts.append(coord_summary)
        
        full_reasoning = "\n".join(reasoning_parts)
        
        # 最终答案生成
        final_prompt_text = "\nBased on the enhanced FOT analysis above, my direct answer is: "
        final_tokens = model.tokenizer.encode(final_prompt_text, add_special_tokens=False, return_tensors='pt')
        
        if final_tokens is not None and final_tokens.numel() > 0:
            final_tokens = final_tokens.to(device, non_blocking=True).long()
            final_batch = {
                'input_ids': torch.cat([current_prompt, final_tokens], dim=1),
                'attention_mask': torch.ones(current_prompt.shape[0], 
                                           current_prompt.shape[1] + final_tokens.shape[1], 
                                           dtype=torch.long, device=device)
            }
            
            final_outputs = safe_generate(
                model,
                input_ids=final_batch['input_ids'],
                attention_mask=final_batch['attention_mask'],
                max_new_tokens=min(max_new_tokens, 100),
                temperature=temperature * 0.7,
                do_sample=False,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id
            )
            
            if final_outputs is not None and len(final_outputs) > 0:
                final_new_tokens = final_outputs[0][final_batch['input_ids'].shape[1]:]
                if final_new_tokens.numel() > 0:
                    final_answer = model.tokenizer.decode(final_new_tokens, skip_special_tokens=True).strip()
                    final_answer = clean_prediction_enhanced(final_answer)
                else:
                    final_answer = "Unable to generate final answer"
            else:
                final_answer = "Generation failed"
        else:
            final_answer = "Token encoding failed"
        
        print(f"[DEBUG] Enhanced FOT completed successfully")
        print(f"[DEBUG] Final answer: {final_answer}")
        print(f"[DEBUG] Coordinate history: {fot_tree['coordinate_history']}")
        
        return {
            'full_reasoning': full_reasoning,
            'final_answer': final_answer,
            'fot_tree': fot_tree,
            'coordinate_history': fot_tree['coordinate_history']
        }, final_answer, reasoning_parts
        
    except Exception as e:
        print(f"[ERROR] Enhanced FOT inference failed: {e}")
        return f"Error: {e}", "Error", []

def evaluate_gqa_with_enhanced_fot(model, preprocessor, gqa_data, use_enhanced_fot=True, 
                                  max_steps=3, coordinate_fusion=True, **kwargs):
    """
    使用增强FOT进行GQA评估
    """
    results = []
    correct = 0
    total = 0
    
    initialize_model_attributes(model)
    device = _get_device_of_model(model)
    data_collator = SFT_DataCollator(tokenizer=preprocessor.tokenizer, sd_tokenizer=None)
    
    if isinstance(gqa_data, dict):
        questions_items = list(gqa_data.items())
    elif isinstance(gqa_data, list):
        questions_items = [(item.get('questionId', str(i)), item) for i, item in enumerate(gqa_data)]
    else:
        raise ValueError("gqa_data must be either a dict or list")
    
    print(f"Starting Enhanced FOT evaluation on {len(questions_items)} questions...")
    
    for question_id, item in tqdm(questions_items, desc="Enhanced FOT Evaluation"):
        try:
            # 数据预处理（与原版相同）
            if isinstance(item, dict) and 'question' in item:
                question = item['question']
                answer = item.get('answer', '')
                image_id = item.get('imageId', '')
            else:
                question = item.get('question', '')
                answer = item.get('answer', '')
                image_id = item.get('imageId', item.get('image', ''))
            
            if not question:
                continue
            
            # 图像加载
            image_path = None
            image_dir = os.environ.get('GQA_IMAGE_DIR', '')
            candidates = [
                os.path.join(image_dir, f"{image_id}.jpg") if image_dir else None,
                image_id,
                f"{image_id}.jpg",
                os.path.join("images", f"{image_id}.jpg"),
                os.path.join("val_images", f"{image_id}.jpg"),
                os.path.join("GQA", "images", f"{image_id}.jpg"),
            ]
            candidates = [p for p in candidates if p is not None]
            for p in candidates:
                if os.path.exists(p):
                    image_path = p
                    break
            
            if image_path is None:
                print(f"Warning: Image not found for {image_id}")
                continue
            
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
            
            # 构造查询
            query = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + \
                   "Please analyze this image step by step and answer the question with coordinate information.\n" + question
            
            conv = [{'from': 'human', 'value': query}]
            item_dict = {'input_images': [image], 'conversation': conv}
            
            # 预处理
            try:
                input_item = preprocessor(item_dict)
                if input_item is None:
                    continue
                
                batch = data_collator([input_item])
                if batch is None or 'input_ids' not in batch:
                    continue
                
                batch = sanitize_batch(batch, preprocessor.tokenizer, device)
                if batch is None:
                    continue
                
            except Exception as e:
                print(f"Preprocessing error for question {question_id}: {e}")
                continue
            
            # 使用增强FOT推理
            if use_enhanced_fot:
                try:
                    detailed_result, prediction, thoughts = enhanced_fot_inference_with_coordination(
                        model, batch,
                        max_steps=max_steps,
                        max_new_tokens=kwargs.get('max_new_tokens', 60),
                        temperature=kwargs.get('temperature', 0.1)
                    )
                    
                    if isinstance(detailed_result, dict):
                        full_reasoning = detailed_result.get('full_reasoning', '')
                        coordinate_info = detailed_result.get('coordinate_history', [])
                    else:
                        full_reasoning = str(detailed_result)
                        coordinate_info = []
                    
                except Exception as e:
                    print(f"Enhanced FOT failed for question {question_id}: {e}")
                    prediction = "Error"
                    full_reasoning = f"Error: {e}"
                    coordinate_info = []
            else:
                # 回退到原始方法
                prediction = "Fallback method not implemented"
                full_reasoning = "Using fallback"
                coordinate_info = []
            
            # 清理预测结果
            if prediction and prediction != "Error":
                prediction = clean_prediction_enhanced(prediction, question)
            
            # 评估准确性
            is_correct = False
            if answer and prediction and prediction != "Error":
                # 简单的字符串匹配评估
                pred_clean = prediction.lower().strip()
                ans_clean = answer.lower().strip()
                is_correct = (pred_clean == ans_clean or 
                            pred_clean in ans_clean or 
                            ans_clean in pred_clean)
                
                if is_correct:
                    correct += 1
            
            total += 1
            
            # 记录结果
            result = {
                'question_id': question_id,
                'question': question,
                'ground_truth': answer,
                'prediction': prediction,
                'is_correct': is_correct,
                'reasoning': full_reasoning,
                'coordinates': coordinate_info,
                'method': 'Enhanced FOT'
            }
            results.append(result)
            
            # 定期报告进度
            if total % 10 == 0:
                accuracy = correct / total if total > 0 else 0
                print(f"Progress: {total} questions, accuracy: {accuracy:.3f}")
        
        except Exception as e:
            print(f"Error processing question {question_id}: {e}")
            continue
    
    # 最终统计
    final_accuracy = correct / total if total > 0 else 0
    print(f"\nEnhanced FOT Evaluation Complete:")
    print(f"Total questions: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {final_accuracy:.3f}")
    
    return {
        'results': results,
        'accuracy': final_accuracy,
        'total': total,
        'correct': correct,
        'method': 'Enhanced FOT with Coordination and Coordinates'
    }
