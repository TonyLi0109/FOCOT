from genericpath import samestat
from torch.utils.data import ConcatDataset, DataLoader
from typing import Optional, Dict
from dataclasses import dataclass, field
from locals.datasets import SFT_DataCollator, WrappedDataset
from lightning.pytorch import seed_everything
from torchvision import transforms
from constants import *
from PIL import Image
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from locals.datasets.preprocessor import VoCoT_InputProcessor
from omegaconf import OmegaConf
from utils.util import instantiate_from_config
from model.language_model.volcano_llama import VolCanoLlamaForCausalLM,VolCanoConfig
from model.language_model.volcano_mistral import VolCanoMistralForCausalLM, VolCanoMistralConfig
from transformers import LlamaTokenizer, AutoTokenizer
import transformers
from peft import PeftConfig, PeftModel
from argparse import ArgumentParser
import os
import torch.distributed as dist
from utils.logger import setup_logger
import json
import tqdm

def rank0_print(args, res):
    if args.local_rank==0 or args.local_rank == -1:
        print(res)

def get_output_name(args, mid_output=True):
    if mid_output:
        return os.path.join(args.output_dir, 
                            '{}_rank{}.json'.format(args.dataset_name, args.local_rank))
    else:
        return os.path.join(args.output_dir, 
                            '{}.json'.format(args.dataset_name))

def get_all_output_names(args):
    return [os.path.join(args.output_dir, 
                            '{}_rank{}.json'.format(args.dataset_name, r)) for r in range(args.n_gpus)]

class CLIPTransform:
    def __init__(self, transform, square_size=None):
        self.transform = transform
        self.square_size = square_size
        self.image_mean = transform.image_mean
    
    def __call__(self, image):
        if self.square_size is not None:
            image = image.resize((self.square_size, self.square_size))
        try:
            tmp = torch.tensor(self.transform(image)['pixel_values'][0])
        except:
            tmp = torch.tensor(self.transform(Image.new(image.mode, (32, 32), (0,0,0)))['pixel_values'][0])
        return tmp



def load_model(model_path, device='cuda:0', precision='bf16'):
    config_class = VolCanoMistralConfig
    model_class = VolCanoMistralForCausalLM
    tokenizer_class = AutoTokenizer
    device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=None,
        use_fast=True,
        trust_remote_code=True
    )
    
    llama_config = config_class.from_pretrained(model_path)
    model = model_class.from_pretrained(model_path, config=llama_config)

    model.input_img_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMG_TOKEN)
    model.eoc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_EOC_TOKEN)
    model.boc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_BOC_TOKEN)
    model.tokenizer = tokenizer
    model.sub_image_bind = False

    if precision == 'bf16':
        model.to(torch.bfloat16)
    elif precision == 'fp16':
        model.to(torch.float16)
    elif precision == 'fp32':
        pass
    else:
        raise ValueError('precision must be fp16, bf16, or fp32')
    model.eval()
    model.to(device)

    resize2square = False
    output_vis_processor = transforms.Compose(
                [
                    transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(1024),
                    # transforms.RandomHorizontalFlip(), # comment here
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
    input_vis_processor = transforms.Compose(
            [
                transforms.Resize((448, 448) if resize2square else 448, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(448),
                # transforms.RandomHorizontalFlip(), comment here
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
        )
    if hasattr(model.vision_encoder, 'image_processor'):
        input_vis_processor = model.vision_encoder.image_processor
        if resize2square:
            tmp_size = input_vis_processor.size['shortest_edge']
        else:
            tmp_size = None
        input_vis_processor = CLIPTransform(input_vis_processor, square_size=tmp_size)
    # tokenizer = LlamaTokenizer.from_pretrained('eval/debug/edit_gpt_emu_tokenizer')

    model.image_processor = None
    preprocessor = VoCoT_InputProcessor(tokenizer=tokenizer, input_image_processor = input_vis_processor, use_mistral=True,
                                                output_image_processor= output_vis_processor, merge_in_out_image=True, expand2square=True, inference = True)

    return model, preprocessor

def infer(model, preprocessor, image, query, cot=True, max_new_tokens=1024, temperature=0.0):
    if cot:
        query = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + query + COT_ACTIVATION
    else:
        query = ALL_IMG_TOKENS_STR + '\n' + query
    conv = [{'from': 'human', 'value':query}]
    item = {'input_images': [image], 'conversation': conv}
    input_item = preprocessor(item)
    data_collator = SFT_DataCollator(tokenizer=preprocessor.tokenizer, sd_tokenizer=None)
    batch = data_collator([input_item])
    txt_res, out_imgs, txt_ids = model.condition_completion(batch, avoid_image_gen=True, 
                                                            max_new_tokens=max_new_tokens, temperature=temperature)

    return txt_res

def infer_with_fot(model, preprocessor, image, query, cot=True, max_new_tokens=1024, 
                   temperature=0.0, use_fot=True, num_thoughts=3):
    """
    带FOT筛选的推理函数
    
    Args:
        model: 加载的volcano模型
        preprocessor: 预处理器
        image: 输入图像
        query: 查询文本
        cot: 是否使用思维链
        max_new_tokens: 最大生成token数
        temperature: 生成温度
        use_fot: 是否启用FOT筛选
        num_thoughts: 生成的思维候选数量
    
    Returns:
        筛选后的最佳回答
    """
    if cot:
        query = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + query + COT_ACTIVATION
    else:
        query = ALL_IMG_TOKENS_STR + '\n' + query
    
    conv = [{'from': 'human', 'value': query}]
    item = {'input_images': [image], 'conversation': conv}
    input_item = preprocessor(item)
    data_collator = SFT_DataCollator(tokenizer=preprocessor.tokenizer, sd_tokenizer=None)
    batch = data_collator([input_item])
    
    # 设置是否使用FOT
    if hasattr(model, 'use_fot'):
        model.use_fot = use_fot
    
    if use_fot and hasattr(model, 'condition_completion_with_fot'):
        txt_res, out_imgs, txt_ids = model.condition_completion_with_fot(
            batch, avoid_image_gen=True, max_new_tokens=max_new_tokens, 
            temperature=temperature, num_thoughts=num_thoughts
        )
    else:
        # 如果模型不支持FOT或未启用，则使用原始方法
        if use_fot:
            print("[Warning] FOT not available, using standard inference")
        txt_res, out_imgs, txt_ids = model.condition_completion(
            batch, avoid_image_gen=True, max_new_tokens=max_new_tokens, 
            temperature=temperature
        )
    
    return txt_res

def batch_infer_with_fot(model, preprocessor, images, queries, cot=True, max_new_tokens=1024,
                        temperature=0.0, use_fot=True, num_thoughts=3):
    """
    批量FOT推理函数
    
    Args:
        model: 加载的volcano模型
        preprocessor: 预处理器
        images: 输入图像列表
        queries: 查询文本列表
        其他参数同infer_with_fot
    
    Returns:
        批量推理结果列表
    """
    results = []
    
    for i, (image, query) in enumerate(zip(images, queries)):
        try:
            result = infer_with_fot(
                model, preprocessor, image, query, 
                cot=cot, max_new_tokens=max_new_tokens,
                temperature=temperature, use_fot=use_fot, 
                num_thoughts=num_thoughts
            )
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(images)} samples")
                
        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            results.append(["Error: Failed to process"])
    
    return results

def infer_with_monte_carlo_fot(model, preprocessor, image, query, cot=True, max_new_tokens=1024,
                              temperature=0.0, use_monte_carlo=True, num_iterations=10,
                              ucb_constant=1.4, max_depth=5, early_stop_threshold=0.8):
    """
    带蒙特卡洛树搜索FOT的推理函数
    
    Args:
        model: 加载的volcano模型
        preprocessor: 预处理器
        image: 输入图像
        query: 查询文本
        cot: 是否使用思维链
        max_new_tokens: 最大生成token数
        temperature: 生成温度
        use_monte_carlo: 是否启用蒙特卡洛树搜索
        num_iterations: MCTS迭代次数
        ucb_constant: UCB常数
        max_depth: 最大搜索深度
        early_stop_threshold: 早停阈值
    
    Returns:
        蒙特卡洛树搜索后的最佳回答和搜索统计信息
    """
    if cot:
        query = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + query + COT_ACTIVATION
    else:
        query = ALL_IMG_TOKENS_STR + '\n' + query
    
    conv = [{'from': 'human', 'value': query}]
    item = {'input_images': [image], 'conversation': conv}
    input_item = preprocessor(item)
    data_collator = SFT_DataCollator(tokenizer=preprocessor.tokenizer, sd_tokenizer=None)
    batch = data_collator([input_item])
    
    # 检查模型是否支持蒙特卡洛FOT
    if use_monte_carlo and hasattr(model, 'run_monte_carlo_inference'):
        # 设置蒙特卡洛参数
        if hasattr(model, 'monte_carlo_forest'):
            model.monte_carlo_forest.num_iterations = num_iterations
            model.monte_carlo_forest.ucb_constant = ucb_constant
            model.monte_carlo_forest.max_depth = max_depth
            model.monte_carlo_forest.early_stop_threshold = early_stop_threshold
        
        # 运行蒙特卡洛推理
        result = model.run_monte_carlo_inference(
            batch, avoid_image_gen=True, max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        if isinstance(result, dict):
            return result['best_response'], result.get('stats', {})
        else:
            return result, {}
    
    elif use_monte_carlo and hasattr(model, 'condition_completion_with_fot'):
        # 回退到简单FOT
        print("[Warning] Monte Carlo FOT not available, using simple FOT")
        txt_res, out_imgs, txt_ids = model.condition_completion_with_fot(
            batch, avoid_image_gen=True, max_new_tokens=max_new_tokens,
            temperature=temperature, num_thoughts=3
        )
        return txt_res, {'fallback_to_simple_fot': True}
    
    else:
        # 回退到标准推理
        if use_monte_carlo:
            print("[Warning] Monte Carlo FOT not available, using standard inference")
        txt_res, out_imgs, txt_ids = model.condition_completion(
            batch, avoid_image_gen=True, max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        return txt_res, {'fallback_to_standard': True}

def batch_infer_with_monte_carlo_fot(model, preprocessor, images, queries, cot=True, 
                                    max_new_tokens=1024, temperature=0.0, use_monte_carlo=True,
                                    num_iterations=10, ucb_constant=1.4, max_depth=5, 
                                    early_stop_threshold=0.8):
    """
    批量蒙特卡洛FOT推理函数
    
    Args:
        model: 加载的volcano模型
        preprocessor: 预处理器
        images: 输入图像列表
        queries: 查询文本列表
        其他参数同infer_with_monte_carlo_fot
    
    Returns:
        批量推理结果列表和统计信息
    """
    results = []
    all_stats = []
    
    for i, (image, query) in enumerate(zip(images, queries)):
        try:
            result, stats = infer_with_monte_carlo_fot(
                model, preprocessor, image, query,
                cot=cot, max_new_tokens=max_new_tokens,
                temperature=temperature, use_monte_carlo=use_monte_carlo,
                num_iterations=num_iterations, ucb_constant=ucb_constant,
                max_depth=max_depth, early_stop_threshold=early_stop_threshold
            )
            results.append(result)
            all_stats.append(stats)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(images)} samples with Monte Carlo FOT")
                
        except Exception as e:
            print(f"Error processing sample {i} with Monte Carlo FOT: {str(e)}")
            results.append(["Error: Failed to process with Monte Carlo FOT"])
            all_stats.append({'error': str(e)})
    
    return results, all_stats

def setup_monte_carlo_fot_model(model, num_iterations=10, ucb_constant=1.4, 
                                max_depth=5, early_stop_threshold=0.8,
                                confidence_threshold=0.7, coherence_threshold=0.6,
                                consistency_threshold=0.6):
    """
    为模型设置蒙特卡洛FOT参数
    
    Args:
        model: volcano模型实例
        num_iterations: MCTS迭代次数
        ucb_constant: UCB常数
        max_depth: 最大搜索深度
        early_stop_threshold: 早停阈值
        confidence_threshold: 置信度阈值
        coherence_threshold: 连贯性阈值
        consistency_threshold: 一致性阈值
    
    Returns:
        配置好的模型
    """
    if hasattr(model, 'monte_carlo_forest'):
        # 设置蒙特卡洛参数
        model.monte_carlo_forest.num_iterations = num_iterations
        model.monte_carlo_forest.ucb_constant = ucb_constant
        model.monte_carlo_forest.max_depth = max_depth
        model.monte_carlo_forest.early_stop_threshold = early_stop_threshold
        
        print(f"Monte Carlo FOT configured with:")
        print(f"  - Iterations: {num_iterations}")
        print(f"  - UCB Constant: {ucb_constant}")
        print(f"  - Max Depth: {max_depth}")
        print(f"  - Early Stop Threshold: {early_stop_threshold}")
    
    if hasattr(model, 'fot_filter'):
        # 设置FOT过滤器参数
        model.fot_filter.confidence_threshold = confidence_threshold
        model.fot_filter.coherence_threshold = coherence_threshold
        model.fot_filter.consistency_threshold = consistency_threshold
        
        print(f"FOT Filter configured with:")
        print(f"  - Confidence Threshold: {confidence_threshold}")
        print(f"  - Coherence Threshold: {coherence_threshold}")
        print(f"  - Consistency Threshold: {consistency_threshold}")
    
    return model

if __name__=='__main__':
    from PIL import Image
    tmp_image = Image.open('eval/debug/tmp.jpg')
    model_path = '/mnt/bn/yangmin-priv/luoruipu/checkpoints/LLaVA-clip336px-obj-represent-Mistral-1e-5-3072-instruct_llava+shikraCoT75per+GPTQTA+lvis-cot/'
    model, preprocessor = load_model(model_path,precision='fp16')
    
    # 配置蒙特卡洛FOT参数
    model = setup_monte_carlo_fot_model(
        model, 
        num_iterations=15,
        ucb_constant=1.4,
        max_depth=5,
        early_stop_threshold=0.8
    )
    
    # 原始推理测试
    print("=== 原始推理测试 ===")
    res1 = infer(model, preprocessor, tmp_image, 'Is there a event "the cat is below the bed" in this image?', cot=True)
    res = infer(model, preprocessor, tmp_image, 'Why is the cat on the bed?', cot=True)
    res_no_cot = infer(model, preprocessor, tmp_image, 'Describe the image.', cot=True)
    print(res1)
    print(res)
    print(res_no_cot)
    
    # FOT推理测试
    print("\n=== FOT推理测试 ===")
    fot_res1 = infer_with_fot(model, preprocessor, tmp_image, 'Is there a event "the cat is below the bed" in this image?', 
                             cot=True, use_fot=True, num_thoughts=3)
    fot_res2 = infer_with_fot(model, preprocessor, tmp_image, 'Why is the cat on the bed?', 
                             cot=True, use_fot=True, num_thoughts=3)
    print("FOT结果1:", fot_res1)
    print("FOT结果2:", fot_res2)
    
    # 蒙特卡洛FOT推理测试
    print("\n=== 蒙特卡洛FOT推理测试 ===")
    mc_res1, mc_stats1 = infer_with_monte_carlo_fot(
        model, preprocessor, tmp_image, 
        'Is there a event "the cat is below the bed" in this image?',
        cot=True, use_monte_carlo=True, num_iterations=10
    )
    mc_res2, mc_stats2 = infer_with_monte_carlo_fot(
        model, preprocessor, tmp_image, 'Why is the cat on the bed?',
        cot=True, use_monte_carlo=True, num_iterations=10
    )
    print("蒙特卡洛FOT结果1:", mc_res1)
    print("蒙特卡洛FOT统计1:", mc_stats1)
    print("蒙特卡洛FOT结果2:", mc_res2)
    print("蒙特卡洛FOT统计2:", mc_stats2)