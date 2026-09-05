from typing import List, Optional, Tuple, Union
from xml.dom.expatbuilder import parseString
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                        LlamaConfig, LlamaModel, LlamaForCausalLM, CLIPImageProcessor
from transformers import MistralForCausalLM, MistralConfig, MistralModel

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput

from ..vision_encoder.builder import build_vision_encoder
from ..front_projector.builder import build_front_projector
from .volcano_base import VolCanoMetaForCausalLM
from transformers import GenerationMixin

from PIL import Image
from constants import *
import random
from utils.util import rank_0_print
import torch.nn.functional as F
from diffusers.models.vae import DiagonalGaussianDistribution
from transformers import StoppingCriteria, StoppingCriteriaList
from utils.eval_util import extract_box_str
from locals.datasets.utils.box_utils import *
from utils.fot_filter import FOTFilter

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False

def resize_image_to_square(image):
    width, height = image.size
    max_side = max(width, height)
    image = image.resize((max_side, max_side))
    return image

def split_tensor_func(tensor, dim=0):
    num_splits = tensor.shape[dim]
    return [f.squeeze(dim) for f in tensor.split([1]*num_splits, dim=dim)]

@dataclass
class VolCanoCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    regression_loss: Optional[torch.FloatTensor] = None
    text_loss: Optional[torch.FloatTensor] = None

IGNORE_INDEX = -100

def disabled_train(self, mode=True):
    return self

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        return super().forward(x)

class VolCanoMistralConfig(MistralConfig):
    model_type = "VolCanoMistral"

class VolCanoMistralModel(MistralModel):
    config_class = VolCanoMistralConfig

    def __init__(self, config: MistralConfig):
        super(VolCanoMistralModel, self).__init__(config)

class VolCanoMistralForCausalLM(MistralForCausalLM, VolCanoMetaForCausalLM):
    config_class = VolCanoMistralConfig

    def __init__(self, config):
        super(MistralForCausalLM, self).__init__(config)
        
        self.model = VolCanoMistralModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.init_vision_model()
        
        # 初始化FOT过滤器
        self.fot_filter = FOTFilter()
        self.use_fot = True
        
        if hasattr(self, 'vision_generator'):
            zero_img_feature = torch.zeros((1, IMG_TOKEN_NUM, config.hidden_size))
            self.register_buffer('zero_img_feature', zero_img_feature, persistent=False)

        self.post_init()
        self.output_img_id = -100
        self.regression_weight = 1.0
        self.no_bind = False
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        captions=[None],
        output_image_feature=None,
        output_images = None,
        output_cond_images = None,
        output_cond_img_mask = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        input_images: Optional[torch.FloatTensor] = None,
        image_label_masks: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        item_id: Optional[bool] = None,
        box: Optional[torch.Tensor] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # ... existing code ...
        # 这里保持原有的forward方法实现不变
        pass
    
    def condition_completion_with_fot(self, input_dict, temperature=0.2, max_new_tokens=128, 
                                     guidance_scale=7.5, avoid_image_gen=False, 
                                     num_thoughts=3, **kwargs):
        """
        使用FOT机制的条件完成方法
        """
        if not self.use_fot or not hasattr(self, 'fot_filter'):
            return self.condition_completion(input_dict, temperature, max_new_tokens, 
                                           guidance_scale, avoid_image_gen, **kwargs)
        
        # 生成多个思维候选
        thoughts = []
        logits_list = []
        
        for i in range(num_thoughts):
            # 添加随机性以生成不同的候选
            temp_temperature = temperature + random.uniform(-0.1, 0.1)
            temp_temperature = max(0.1, min(1.0, temp_temperature))
            
            result = self.condition_completion(
                input_dict, 
                temperature=temp_temperature, 
                max_new_tokens=max_new_tokens,
                guidance_scale=guidance_scale, 
                avoid_image_gen=avoid_image_gen, 
                **kwargs
            )
            
            if isinstance(result, tuple):
                thought, logits = result[0], result[1] if len(result) > 1 else None
            else:
                thought, logits = result, None
                
            thoughts.append(thought)
            if logits is not None:
                logits_list.append(logits)
        
        # 使用FOT过滤器筛选最佳思维
        question = input_dict.get('question', '')
        filtered_thoughts = self.fot_filter.filter_multiple_thoughts(
            thoughts, question, logits_list if logits_list else None
        )
        
        if filtered_thoughts:
            # 返回质量最高的思维
            best_thought, score = filtered_thoughts[0]
            return best_thought
        else:
            # 如果没有思维通过筛选，返回第一个候选
            return thoughts[0]
    
    def calculate_options_with_fot(self, input_dict, cot=False, further_instruct=False, 
                                  temperature=0.2, max_new_tokens=128, guidance_scale=7.5, 
                                  avoid_image_gen=False, likelihood_reduction='mean', 
                                  num_thoughts=3, **kwargs):
        """
        使用FOT机制的选项计算方法
        """
        if not self.use_fot or not hasattr(self, 'fot_filter'):
            return self.calculate_options(input_dict, cot, further_instruct, temperature, 
                                        max_new_tokens, guidance_scale, avoid_image_gen, 
                                        likelihood_reduction, **kwargs)
        
        # 生成多个思维候选
        thoughts = []
        for i in range(num_thoughts):
            temp_temperature = temperature + random.uniform(-0.1, 0.1)
            temp_temperature = max(0.1, min(1.0, temp_temperature))
            
            result = self.calculate_options(
                input_dict, cot, further_instruct, temp_temperature,
                max_new_tokens, guidance_scale, avoid_image_gen, 
                likelihood_reduction, **kwargs
            )
            thoughts.append(result)
        
        # 使用FOT过滤器筛选
        question = input_dict.get('question', '')
        thought_texts = [str(thought) for thought in thoughts]
        filtered_thoughts = self.fot_filter.filter_multiple_thoughts(thought_texts, question)
        
        if filtered_thoughts:
            # 找到对应的原始结果
            best_thought_text = filtered_thoughts[0][0]
            for i, thought_text in enumerate(thought_texts):
                if thought_text == best_thought_text:
                    return thoughts[i]
        
        # 如果没有通过筛选，返回第一个
        return thoughts[0]

    # ... existing code ...
    # 保持其他所有原有方法不变
    
AutoConfig.register("VolCanoMistral", VolCanoMistralConfig)
AutoModelForCausalLM.register(VolCanoMistralConfig, VolCanoMistralForCausalLM)