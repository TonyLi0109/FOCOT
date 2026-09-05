# -*- coding: utf-8 -*-
"""
AMBER数据集的FOT评估实现
基于evaluate_benchmark.py的标准流程，确保与现有基础设施兼容
"""
import json
import os
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
import math
import re
from typing import Any, Dict, List, Tuple, Optional
from contextlib import nullcontext

from locals.datasets import SFT_DataCollator
from locals.datasets.preprocessor import VoCoT_InputProcessor
from locals.datasets import WrappedDataset
from locals.datasets.eval.short_qa import AMBERDataset
from constants import ALL_IMG_TOKENS_STR, DEFAULT_GRD_TOKEN

# AMBER特定配置
AMBER_ACTIVATION = 'Please answer the question based on the image. Provide a clear and direct text answer.'
MAX_INPUT_LEN = 4096
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.1

# -------------------- 基础工具函数 --------------------
def _get_device_of_model(model: torch.nn.Module) -> torch.device:
    """获取模型所在设备"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')

def _to_device(obj, device: torch.device):
    """将对象移动到指定设备"""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj

def _is_cuda_assert(e: Exception) -> bool:
    """检查是否为CUDA断言错误"""
    msg = str(e).lower()
    return ("device-side assert" in msg) or ("device side assert" in msg) or ("indexselectlargeindex" in msg)

def validate_and_fix_tokens(input_ids, tokenizer, max_length=MAX_INPUT_LEN):
    """验证和修复token序列，防止CUDA索引错误"""
    if input_ids is None:
        return None
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
    if input_ids.dtype != torch.long:
        input_ids = input_ids.long()

    # 长度裁剪
    if input_ids.shape[-1] > max_length:
        input_ids = input_ids[..., :max_length]

    # 词表范围检查
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
            if _is_cuda_assert(e):
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
        if _is_cuda_assert(e):
            print(f"Warning: CUDA assert in token validation, skipping validation")
        else:
            raise e

    return input_ids

def sanitize_batch(batch, tokenizer, device):
    """清理和验证batch，防止CUDA错误"""
    if batch is None:
        return None
    
    try:
        # 验证和修复input_ids
        if 'input_ids' in batch:
            input_ids = batch['input_ids']
            if isinstance(input_ids, torch.Tensor):
                # 验证和修复tokens
                input_ids = validate_and_fix_tokens(input_ids, tokenizer)
                if input_ids is None:
                    return None
                batch['input_ids'] = input_ids.to(device, non_blocking=True)
        
        # 确保attention_mask存在且正确
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

    return batch

def safe_generate(model, **kwargs):
    """安全的生成函数，处理各种异常情况包括CUDA错误"""
    try:
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 验证输入参数
        if 'input_ids' not in kwargs:
            return {
                'success': False,
                'outputs': None,
                'error': "Missing input_ids"
            }
        
        input_ids = kwargs['input_ids']
        if not isinstance(input_ids, torch.Tensor):
            return {
                'success': False,
                'outputs': None,
                'error': "input_ids must be a tensor"
            }
        
        # 检查input_ids的形状和值范围
        if len(input_ids.shape) != 2:
            return {
                'success': False,
                'outputs': None,
                'error': f"input_ids shape is {input_ids.shape}, expected 2D tensor"
            }
        
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
        try:
            invalid_tokens = (input_ids >= vocab_size) | (input_ids < 0)
            if invalid_tokens.any():
                print(f"Warning: Found {invalid_tokens.sum().item()} invalid tokens")
                # 将无效token替换为pad_token_id
                input_ids = input_ids.clone()
                input_ids[invalid_tokens] = model.tokenizer.pad_token_id
                kwargs['input_ids'] = input_ids
        except RuntimeError as e:
            if _is_cuda_assert(e):
                print(f"Warning: CUDA assert in token validation, skipping")
            else:
                raise e
        
        # 设置默认参数
        default_kwargs = {
            'max_new_tokens': min(kwargs.get('max_new_tokens', DEFAULT_MAX_NEW_TOKENS), 256),
            'temperature': DEFAULT_TEMPERATURE,
            'do_sample': False,
            'use_cache': False,
            'pad_token_id': getattr(model.tokenizer, 'pad_token_id', 0),
            'eos_token_id': getattr(model.tokenizer, 'eos_token_id', 2),
        }
        
        # 合并参数
        for k, v in default_kwargs.items():
            if k not in kwargs:
                kwargs[k] = v
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(**kwargs)
        
        return {
            'success': True,
            'outputs': outputs,
            'error': None
        }
    except RuntimeError as e:
        if _is_cuda_assert(e):
            print(f"CUDA assert detected in generation: {e}")
            # 尝试CPU回退
            try:
                print("Attempting CPU fallback...")
                device = torch.device('cpu')
                model.to(device)
                
                # 只移动tensor类型的参数到CPU，跳过其他类型
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        kwargs[k] = v.to(device)
                    # 跳过非tensor类型的参数（如bool, int, float等）
                
                with torch.no_grad():
                    outputs = model.generate(**kwargs)
                
                return {
                    'success': True,
                    'outputs': outputs,
                    'error': None
                }
            except Exception as cpu_e:
                print(f"CPU fallback also failed: {cpu_e}")
                return {
                    'success': False,
                    'outputs': None,
                    'error': f"CUDA error with failed CPU fallback: {str(e)}"
                }
        else:
            print(f"Generation error: {e}")
            return {
                'success': False,
                'outputs': None,
                'error': str(e)
            }
    except Exception as e:
        print(f"Generation error: {e}")
        return {
            'success': False,
            'outputs': None,
            'error': str(e)
        }

# -------------------- FOT核心功能 --------------------
def clean_prediction(prediction):
    """清理预测结果，移除多余的前缀和格式化"""
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

def initialize_model_attributes(model):
    """初始化模型属性，确保FOT推理所需的属性存在"""
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

def step_by_step_inference(model, batch, max_steps=3, max_new_tokens=100, temperature=0.1):
    """
    FOT逐步推理实现 - 安全版本，包含CUDA错误处理
    """
    # 初始化模型属性
    initialize_model_attributes(model)
    
    device = _get_device_of_model(model)
    
    # 清理和验证batch
    batch = sanitize_batch(batch, model.tokenizer, device)
    if batch is None:
        return {
            'success': False,
            'thoughts': [],
            'final_answer': "",
            'error': "Failed to sanitize batch"
        }
    
    thoughts = []
    
    # 获取原始输入的input_ids作为基础
    if 'input_ids' not in batch:
        return {
            'success': False,
            'thoughts': [],
            'final_answer': "",
            'error': "No input_ids in batch"
        }
    
    current_prompt = batch['input_ids'].clone().to(device, non_blocking=True)
    if current_prompt.dtype != torch.long:
        current_prompt = current_prompt.long()
    
    try:
        for step in range(max_steps):
            print(f"FOT Step {step + 1}/{max_steps}")
            
            # 构建当前步骤的提示
            step_text = f"\nStep {step + 1}: Let me think about this step by step."
            try:
                step_tokens = model.tokenizer.encode(step_text, add_special_tokens=False, return_tensors='pt')
                if step_tokens is None or step_tokens.numel() == 0:
                    continue
                step_tokens = step_tokens.to(device, non_blocking=True).long()
            except Exception as e:
                print(f"Error encoding step text: {e}")
                continue
            
            # 检查序列长度，防止过长
            total_length = current_prompt.shape[1] + step_tokens.shape[1]
            if total_length > MAX_INPUT_LEN:
                print(f"Warning: sequence too long ({total_length}), truncating")
                # 截断current_prompt以留出空间给step_tokens
                max_prompt_len = MAX_INPUT_LEN - step_tokens.shape[1] - max_new_tokens
                if max_prompt_len > 0:
                    current_prompt = current_prompt[:, :max_prompt_len]
                else:
                    print("Sequence too long even after truncation, skipping step")
                    break
            
            # 构建当前批次 - 只包含文本相关的字段
            try:
                current_batch = {
                    'input_ids': torch.cat([current_prompt, step_tokens], dim=1),
                    'attention_mask': torch.ones(current_prompt.shape[0], current_prompt.shape[1] + step_tokens.shape[1], 
                                                dtype=torch.long, device=device)
                }
            except Exception as e:
                print(f"Error creating current batch: {e}")
                break
            
            # 使用安全生成函数，限制生成长度
            gen_result = safe_generate(
                model,
                input_ids=current_batch['input_ids'],
                attention_mask=current_batch['attention_mask'],
                max_new_tokens=min(max_new_tokens, 50),  # 限制每步最多50个token
                temperature=temperature,
                do_sample=temperature > 0,
                # 添加停止条件，防止生成过多步骤
                eos_token_id=model.tokenizer.eos_token_id,
                pad_token_id=model.tokenizer.pad_token_id
            )
            
            if not gen_result['success']:
                print(f"Generation failed: {gen_result['error']}")
                break
            
            outputs = gen_result['outputs']
            # 修复：outputs本身就是生成的序列tensor，不需要.sequences
            if outputs is not None and len(outputs) > 0:
                try:
                    # outputs是tensor，直接使用
                    new_tokens = outputs[0][current_batch['input_ids'].shape[1]:]
                    print(f"[DEBUG] Generated {new_tokens.numel()} new tokens")
                    
                    if new_tokens.numel() > 0:
                        # 打印原始token IDs用于调试
                        print(f"[DEBUG] New token IDs: {new_tokens.tolist()[:10]}...")  # 只显示前10个
                        
                        thought = model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                        print(f"[DEBUG] Decoded thought: '{thought}'")
                        
                        # 清理HTML标签和多余的步骤标记
                        thought = re.sub(r'<[^>]+>', '', thought).strip()
                        
                        # 移除模型自己生成的步骤标记，只保留实际思考内容
                        thought = re.sub(r'Step\s+\d+:\s*', '', thought, flags=re.IGNORECASE).strip()
                        
                        # 如果包含多个步骤，只取第一个有意义的句子
                        if 'Step' in thought:
                            # 找到第一个"Step"之前的内容
                            parts = thought.split('Step')
                            if parts[0].strip():
                                thought = parts[0].strip()
                            else:
                                # 如果第一部分为空，取第一个步骤的内容
                                if len(parts) > 1:
                                    first_step = parts[1]
                                    # 移除步骤编号
                                    first_step = re.sub(r'^\d+:\s*', '', first_step).strip()
                                    # 取第一句话
                                    sentences = [s.strip() for s in first_step.split('.') if s.strip()]
                                    thought = sentences[0] if sentences else ""
                        
                        # 进一步清理，取第一句有意义的话
                        sentences = [s.strip() for s in thought.split('.') if s.strip()]
                        if sentences:
                            thought = sentences[0]
                        
                        print(f"[DEBUG] Cleaned thought: '{thought}'")
                        
                        if thought and len(thought) > 5:  # 只有当思考内容不为空且有意义时才添加
                            thoughts.append({
                                'step': step + 1,
                                'thought': thought,
                                'confidence': 0.8  # 简化的置信度
                            })
                            print(f"[DEBUG] Added thought for step {step + 1}: '{thought}'")
                        else:
                            print(f"[DEBUG] Empty or too short thought after cleaning for step {step + 1}")
                        
                        # 更新当前提示，包含这一步的思考
                        try:
                            if thought and len(thought) > 5:  # 只有当思考内容不为空且有意义时才更新prompt
                                thought_tokens = model.tokenizer.encode(f" {thought}.", add_special_tokens=False, return_tensors='pt')
                                if thought_tokens is not None and thought_tokens.numel() > 0:
                                    thought_tokens = thought_tokens.to(device, non_blocking=True).long()
                                    current_prompt = torch.cat([current_prompt, step_tokens, thought_tokens], dim=1)
                                    print(f"[DEBUG] Updated prompt length: {current_prompt.shape[1]}")
                        except Exception as e:
                            print(f"Error updating prompt: {e}")
                            # 继续下一步，不更新prompt
                    else:
                        print(f"[DEBUG] No new tokens generated for step {step + 1}")
                except Exception as e:
                    print(f"Error processing generation output: {e}")
                    break
            else:
                print(f"[DEBUG] No valid outputs for step {step + 1}")
            
            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 生成最终答案，基于所有思考步骤
        if thoughts:
            # 构建最终答案提示
            final_prompt_text = "\nBased on my step-by-step analysis, the answer is:"
            try:
                final_tokens = model.tokenizer.encode(final_prompt_text, add_special_tokens=False, return_tensors='pt')
                if final_tokens is not None and final_tokens.numel() > 0:
                    final_tokens = final_tokens.to(device, non_blocking=True).long()
                    
                    final_batch = {
                        'input_ids': torch.cat([current_prompt, final_tokens], dim=1),
                        'attention_mask': torch.ones(current_prompt.shape[0], current_prompt.shape[1] + final_tokens.shape[1], 
                                                    dtype=torch.long, device=device)
                    }
                    
                    # 生成最终答案
                    final_result = safe_generate(
                        model,
                        input_ids=final_batch['input_ids'],
                        attention_mask=final_batch['attention_mask'],
                        max_new_tokens=30,  # 最终答案应该简短
                        temperature=0.0,  # 使用贪婪解码获得确定性答案
                        do_sample=False
                    )
                    
                    if final_result['success'] and final_result['outputs'] is not None:
                        final_outputs = final_result['outputs']
                        if len(final_outputs) > 0:
                            final_new_tokens = final_outputs[0][final_batch['input_ids'].shape[1]:]
                            if final_new_tokens.numel() > 0:
                                final_answer = model.tokenizer.decode(final_new_tokens, skip_special_tokens=True).strip()
                                final_answer = re.sub(r'<[^>]+>', '', final_answer).strip()
                                # 取第一句话作为最终答案
                                sentences = [s.strip() for s in final_answer.split('.') if s.strip()]
                                if sentences:
                                    final_answer = sentences[0]
                                print(f"[DEBUG] Generated final answer: '{final_answer}'")
                            else:
                                final_answer = thoughts[-1]['thought']
                        else:
                            final_answer = thoughts[-1]['thought']
                    else:
                        final_answer = thoughts[-1]['thought']
                else:
                    final_answer = thoughts[-1]['thought']
            except Exception as e:
                print(f"Error generating final answer: {e}")
                final_answer = thoughts[-1]['thought']
        else:
            final_answer = ""
        
        print(f"[DEBUG] Final answer: '{final_answer}'")
        print(f"[DEBUG] Total thoughts collected: {len(thoughts)}")
        
        return {
            'success': True,
            'thoughts': thoughts,
            'final_answer': final_answer,
            'error': None
        }
        
    except Exception as e:
        print(f"FOT inference error: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            'success': False,
            'thoughts': thoughts,
            'final_answer': "",
            'error': str(e)
        }

# -------------------- 主要评估函数 --------------------
def evaluate_amber_with_fot_standard(model, preprocessor, dataset, data_collator, 
                                    use_fot=True, num_thoughts=3, 
                                    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                                    temperature=DEFAULT_TEMPERATURE,
                                    avoid_image_gen=True,
                                    **kwargs):
    """
    使用标准evaluate_benchmark.py流程的AMBER FOT评估
    """
    device = _get_device_of_model(model)
    
    # 创建WrappedDataset和DataLoader，完全模拟evaluate_benchmark.py
    from torch.utils.data import DataLoader, SequentialSampler
    
    wrapped_dataset = WrappedDataset('eval', dataset, preprocessor)
    sampler = SequentialSampler(dataset)
    dl = DataLoader(wrapped_dataset, sampler=sampler, batch_size=1, collate_fn=data_collator)
    
    print(f"***** Running AMBER FOT Evaluation *****")
    print(f"  Num examples = {len(dataset)}")
    print(f"  Use FOT = {use_fot}")
    print(f"  Num thoughts = {num_thoughts}")
    
    current_res = []
    
    with torch.inference_mode(), torch.no_grad():
        for batch in tqdm(dl, desc='Evaluating AMBER with FOT'):
            try:
                if use_fot:
                    # FOT推理
                    inference_result = step_by_step_inference(
                        model, batch, 
                        max_steps=num_thoughts, 
                        max_new_tokens=max_new_tokens,
                        temperature=temperature
                    )
                    
                    if inference_result['success']:
                        prediction = inference_result['final_answer']
                        thoughts = inference_result['thoughts']
                        print(f"[DEBUG] FOT inference successful for {batch['item_id'][0]}")
                        print(f"[DEBUG] Raw final answer: '{prediction}'")
                        print(f"[DEBUG] Number of thoughts: {len(thoughts)}")
                    else:
                        print(f"FOT inference failed for {batch['item_id'][0]}: {inference_result['error']}")
                        prediction = ""
                        thoughts = []
                else:
                    # 标准推理 - 使用condition_completion处理raw_images
                    txt_res, out_imgs, txt_ids = model.condition_completion(
                        batch, 
                        avoid_image_gen=avoid_image_gen,
                        max_new_tokens=max_new_tokens, 
                        temperature=temperature
                    )
                    prediction = txt_res[0]
                    thoughts = []
                
                # 清理预测结果
                cleaned_prediction = clean_prediction(prediction)
                print(f"[DEBUG] Prediction before cleaning: '{prediction}'")
                print(f"[DEBUG] Prediction after cleaning: '{cleaned_prediction}'")
                
                # 构建结果字典 - 与evaluate_benchmark.py格式完全一致
                tmp_dict = {
                    'item_id': batch['item_id'][0],
                    'prediction': cleaned_prediction
                }
                
                # 添加FOT特有信息
                if use_fot and thoughts:
                    tmp_dict['thoughts'] = thoughts
                    tmp_dict['num_thoughts'] = len(thoughts)
                    print(f"[DEBUG] Added {len(thoughts)} thoughts to result")
                
                print(f"[DEBUG] Final result dict: {tmp_dict}")
                
                # 添加label信息（AMBER使用getlabel方法）
                if hasattr(dataset, 'getlabel'):
                    try:
                        item_id = int(tmp_dict['item_id'].split('_')[-1])
                        tmp_dict['label'] = dataset.getlabel(item_id)
                    except:
                        pass
                
                # 添加dataset_id信息（如果需要）
                if hasattr(dataset, 'get_index'):
                    try:
                        item_id = int(tmp_dict['item_id'].split('_')[-1])
                        tmp_dict['dataset_id'] = dataset.get_index(item_id)
                    except:
                        pass
                
                current_res.append(tmp_dict)
                
            except Exception as e:
                print(f'Failed to predict {batch["item_id"][0]}: {e}')
                # 创建失败的结果条目
                tmp_dict = {
                    'item_id': batch['item_id'][0],
                    'prediction': ''
                }
                if use_fot:
                    tmp_dict['thoughts'] = []
                    tmp_dict['num_thoughts'] = 0
                
                # 尝试添加label
                if hasattr(dataset, 'getlabel'):
                    try:
                        item_id = int(batch['item_id'][0].split('_')[-1])
                        tmp_dict['label'] = dataset.getlabel(item_id)
                    except:
                        pass
                
                current_res.append(tmp_dict)
            
            # 定期清理内存
            if len(current_res) % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    return current_res

def save_amber_results(results: List[Dict], output_path: str, use_fot: bool = False):
    """
    保存AMBER评估结果
    
    Args:
        results: 评估结果列表
        output_path: 输出文件路径
        use_fot: 是否使用了FOT
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存完整结果
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    print(f"Total questions processed: {len(results)}")
    
    # 统计信息
    successful_predictions = sum(1 for r in results if r.get('prediction', '').strip())
    print(f"Successful predictions: {successful_predictions}/{len(results)}")
    
    if use_fot:
        # FOT特定统计
        total_thoughts = sum(r.get('num_thoughts', 0) for r in results)
        avg_thoughts = total_thoughts / len(results) if results else 0
        print(f"Average thoughts per question: {avg_thoughts:.2f}")
    
    # 提示转换命令
    amber_format_path = output_path.replace('.json', '_amber_format.json')
    print(f"\nTo convert to AMBER evaluation format, run:")
    print(f"python eval/eval_tools/convert_res_to_amber.py --src {output_path} --tgt {amber_format_path} --desc")

# -------------------- 兼容性包装函数 --------------------
def evaluate_amber_dataset_with_fot(model, preprocessor, amber_data, output_path=None,
                                   use_fot=True, num_thoughts=3, 
                                   max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                                   temperature=DEFAULT_TEMPERATURE,
                                   **kwargs):
    """
    AMBER数据集FOT评估的兼容性包装函数
    
    Args:
        model: 模型实例
        preprocessor: 预处理器
        amber_data: AMBER数据集实例或数据路径
        output_path: 输出路径
        use_fot: 是否使用FOT
        num_thoughts: 思维步骤数量
        max_new_tokens: 最大生成token数
        temperature: 生成温度
        **kwargs: 其他参数
    
    Returns:
        Dict: 包含评估结果和统计信息的字典
    """
    from locals.datasets import SFT_DataCollator
    
    # 确保amber_data是数据集实例
    if not hasattr(amber_data, '__len__') or not hasattr(amber_data, '__getitem__'):
        raise ValueError("amber_data must be a dataset instance with __len__ and __getitem__ methods")
    
    # 创建数据整理器
    data_collator = SFT_DataCollator(tokenizer=model.tokenizer, sd_tokenizer=None)
    
    # 运行评估
    results = evaluate_amber_with_fot_standard(
        model=model,
        preprocessor=preprocessor,
        dataset=amber_data,
        data_collator=data_collator,
        use_fot=use_fot,
        num_thoughts=num_thoughts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        **kwargs
    )
    
    # 保存结果
    if output_path:
        save_amber_results(results, output_path, use_fot)
    
    # 计算统计信息
    total_questions = len(results)
    successful_predictions = sum(1 for r in results if r.get('prediction', '').strip())
    accuracy = successful_predictions / total_questions if total_questions > 0 else 0
    
    return {
        'results': results,
        'total_questions': total_questions,
        'successful_predictions': successful_predictions,
        'accuracy': accuracy,
        'use_fot': use_fot,
        'num_thoughts': num_thoughts if use_fot else 0
    }

# -------------------- 向后兼容 --------------------
# 为了与现有代码兼容，保留原函数名
def evaluate_amber_with_fot(model, preprocessor, amber_data, **kwargs):
    """向后兼容的函数名"""
    return evaluate_amber_dataset_with_fot(model, preprocessor, amber_data, **kwargs)