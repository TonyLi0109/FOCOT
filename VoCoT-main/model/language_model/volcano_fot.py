from typing import List, Optional, Tuple, Union
from xml.dom.expatbuilder import parseString
from dataclasses import dataclass
import copy
import math
import numpy as np
from collections import Counter

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

class MonteCarloForest:
    """
    蒙特卡洛森林 - FOT的核心实现
    基于Forest-of-Thought论文的蒙特卡洛树搜索算法
    """
    
    def __init__(self, model, max_iter=2, tree_nums=2, stop_strategy='scaling'):
        self.model = model
        self.max_iter = max_iter
        self.tree_nums = tree_nums
        self.stop_strategy = stop_strategy
        self.correct_num = 0
        self.trees_ans = []
        
    def generate_response(self, prompt, history=[], truncate=True):
        """生成模型响应"""
        # 使用模型生成响应
        input_dict = {'prompt': prompt, 'history': history}
        response = self.model.condition_completion(input_dict, temperature=0.2, max_new_tokens=128)
        return response, history + [prompt, response]
    
    def calculate_reward(self, question, answer):
        """计算答案质量奖励"""
        # 使用FOT过滤器计算质量分数
        if hasattr(self.model, 'fot_filter'):
            passed, score = self.model.fot_filter.filter_thought(answer, question)
            return score * 100  # 转换为0-100分数
        else:
            # 简单的长度和关键词基础评分
            score = min(len(answer.split()) / 20.0, 1.0) * 50
            if any(word in answer.lower() for word in ['yes', 'no', 'true', 'false']):
                score += 25
            return score
    
    def get_weak_answer(self, question, ans_format=''):
        """获取初始弱答案"""
        query = f'Question: {question}\nThe response should begin with [reasoning process]...[Verification]... and end with {ans_format}\nLet\'s think step by step.'
        ans = self.generate_response(query)[0]
        prompt = question + ans + "\nTherefore, the answer is"
        return self.generate_response(prompt)
    
    def get_weak_hints(self, question, weak_answer, history=[]):
        """获取改进提示"""
        query = f'Question: {question}\nSince we have a weak Answer, could you provide me with a reflection or feedback to correct this answer better? Analyze this Answer Strictly and Critic, point out every flaw for every possible imperfect to minus every possible score!\nLet\'s think step by step.'
        return self.generate_response(query, history)
    
    def get_better_answer(self, question, weak_answer, hint, history=[]):
        """获取改进答案"""
        query = f'Question: {question}\nPlease refine your answer according to your Reflection or Feedback. The response should begin with [reasoning process]...[Verification]... and end with the final answer\nLet\'s think step by step.'
        ans = self.generate_response(query, history)[0]
        prompt = question + ans + "\nTherefore, the answer is"
        return self.generate_response(prompt)
    
    def filter_mature_node(self, childs, to_explore, to_explore_reward, max_expand=3):
        """过滤成熟节点"""
        filtered_to_explore = []
        avg_reward = {node: (min(to_explore_reward[node]) + np.mean(to_explore_reward[node])) / 2 
                     for node in to_explore}
        
        for node in to_explore:
            if (len(childs.get(node, [])) < max_expand or 
                max([avg_reward.get(child, -999) for child in childs.get(node, [])]) < avg_reward.get(node, -999)):
                filtered_to_explore.append(node)
        
        return filtered_to_explore
    
    def compute_ucb(self, r_c, N_n, N_c, C=1.4):
        """计算UCB值"""
        return r_c + C * math.sqrt(math.log(N_n + 1) / (N_c + 1e-5))
    
    def update_ucb(self, fathers, childs, to_explore, to_explore_reward, ucb_bank, C=1.4):
        """更新UCB值"""
        # 计算访问次数和平均奖励
        visit_count = {node: len(to_explore_reward[node]) for node in to_explore}
        avg_reward = {node: (min(to_explore_reward[node]) + np.mean(to_explore_reward[node])) / 2 
                     for node in to_explore}
        
        # 获取叶子节点
        leaves = set(to_explore) - set(fathers.values())
        
        # 更新叶子节点UCB值
        for leaf in leaves:
            ucb_bank[leaf] = self.compute_ucb(
                avg_reward[leaf],
                len(to_explore_reward.get(fathers.get(leaf, None), [])),
                len(to_explore_reward.get(leaf, [])),
                C
            )
        
        # 向上更新父节点UCB值
        nodes_to_update = list(leaves)
        while nodes_to_update:
            new_nodes_to_update = set()
            for node in nodes_to_update:
                father = fathers.get(node)
                if father is not None:
                    if father not in ucb_bank:
                        new_nodes_to_update.add(father)
                    if father in ucb_bank:
                        child_reward = [avg_reward[child] for child in childs[father]]
                        if child_reward:
                            father_reward = (avg_reward[father] + max(child_reward)) / 2
                        else:
                            father_reward = avg_reward[father]
                        ucb_bank[father] = self.compute_ucb(
                            father_reward,
                            len(to_explore_reward.get(fathers.get(father, None), [])),
                            len(to_explore_reward.get(father, [])),
                            C
                        )
            nodes_to_update = list(new_nodes_to_update)
    
    def get_best_explore_from_ucb(self, to_explore, ucb_bank):
        """从UCB值中选择最佳探索节点"""
        best_node = None
        highest_ucb = float('-inf')
        
        for node in to_explore:
            ucb_value = ucb_bank.get(node, float('-inf'))
            if ucb_value > highest_ucb:
                highest_ucb = ucb_value
                best_node = node
        
        return best_node
    
    def monte_carlo_tree(self, question, ground_truth=None, max_iter=2):
        """蒙特卡洛树搜索"""
        is_correct = False
        
        # 初始化数据结构
        to_explore = []
        to_explore_reward = {}
        history_bank = {}
        hints_bank = {}
        ucb_bank = {}
        fathers = {}
        childs = {}
        
        def sampling_reward(answer):
            if answer not in to_explore_reward:
                to_explore_reward[answer] = []
            reward = self.calculate_reward(question, answer)
            to_explore_reward[answer].append(reward)
        
        def add_to_hints_bank(hints, weak_answer):
            if weak_answer not in hints_bank:
                hints_bank[weak_answer] = []
            hints_bank[weak_answer].append(hints)
        
        def add_to_childs(father, child):
            if father not in childs:
                childs[father] = []
            childs[father].append(child)
        
        # 获取初始弱答案
        weak_answer, history = self.get_weak_answer(question)
        history_bank[weak_answer] = tuple(history)
        answers_list = [weak_answer]
        to_explore = [weak_answer]
        childs[weak_answer] = []
        fathers[weak_answer] = None
        
        # 添加完全错误答案作为对比
        total_bad = random.choice(["I Don't Know", "I can't understand this question.", 
                                  "I can't help with this question."])
        total_bad_history = copy.deepcopy(history)
        total_bad_history[-1] = total_bad
        history_bank[total_bad] = tuple(total_bad_history)
        answers_list += [total_bad]
        to_explore += [total_bad]
        childs[total_bad] = []
        fathers[total_bad] = None
        
        # 计算初始奖励
        sampling_reward(weak_answer)
        sampling_reward(total_bad)
        
        # 检查初始答案是否正确
        if ground_truth and self._check_answer(ground_truth, weak_answer):
            is_correct = True
            return is_correct, answers_list, to_explore, to_explore_reward, hints_bank, history_bank, fathers, childs, ucb_bank
        
        # 更新UCB值
        self.update_ucb(fathers, childs, to_explore, to_explore_reward, ucb_bank)
        
        # 蒙特卡洛树搜索迭代
        for i in range(max_iter):
            print(f'FOT iteration: {i}')
            
            # 过滤成熟节点
            filtered_to_explore = self.filter_mature_node(childs, to_explore, to_explore_reward)
            
            # 选择最佳探索节点
            weak_answer = self.get_best_explore_from_ucb(filtered_to_explore, ucb_bank)
            
            # 重新采样奖励
            sampling_reward(weak_answer)
            
            # 生成改进提示和答案
            hints, history = self.get_weak_hints(question, weak_answer, history=list(history_bank[weak_answer]))
            answer, history = self.get_better_answer(question, weak_answer, hints, history=history)
            
            # 更新数据结构
            add_to_hints_bank(hints, weak_answer)
            history_bank[answer] = tuple(history)
            to_explore.append(answer)
            sampling_reward(answer)
            fathers[answer] = weak_answer
            childs[answer] = []
            add_to_childs(weak_answer, answer)
            answers_list.append(answer)
            
            # 检查是否找到正确答案
            if ground_truth and self._check_answer(ground_truth, answer):
                is_correct = True
                return is_correct, answers_list, to_explore, to_explore_reward, hints_bank, history_bank, fathers, childs, ucb_bank
            
            # 更新UCB值
            self.update_ucb(fathers, childs, to_explore, to_explore_reward, ucb_bank)
        
        return is_correct, answers_list, to_explore, to_explore_reward, hints_bank, history_bank, fathers, childs, ucb_bank
    
    def _check_answer(self, ground_truth, answer):
        """检查答案是否正确"""
        # 简单的字符串匹配检查
        if isinstance(ground_truth, str) and isinstance(answer, str):
            return ground_truth.lower().strip() in answer.lower().strip()
        return False
    
    def run_forest(self, question, ground_truth=None):
        """运行蒙特卡洛森林"""
        tree_results = []
        
        for t in range(self.tree_nums):
            print(f"Running tree {t + 1}/{self.tree_nums}")
            
            # 为不同的树添加不同的思考方式
            if t > 0:
                # 慢思考模式 - 添加示例引导
                enhanced_question = f"Let me think about this step by step.\nQuestion: {question}"
            else:
                # 快思考模式
                enhanced_question = question
            
            # 运行蒙特卡洛树搜索
            max_iter = min(t + 1, self.max_iter)
            is_correct, answers_list, to_explore, to_explore_reward, hints_bank, history_bank, fathers, childs, ucb_bank = self.monte_carlo_tree(
                enhanced_question, ground_truth, max_iter
            )
            
            # 保存树结果
            tree_data = {
                'tree_num': t,
                'question': enhanced_question,
                'ground_truth': ground_truth,
                'answers_list': answers_list,
                'to_explore': to_explore,
                'to_explore_reward': to_explore_reward,
                'hints_bank': hints_bank,
                'history_bank': history_bank,
                'fathers': fathers,
                'childs': childs,
                'ucb_bank': ucb_bank,
                'is_correct': is_correct
            }
            tree_results.append(tree_data)
            
            # 更新森林答案列表
            self.trees_ans.extend(answers_list)
            
            # 早停策略
            if is_correct and self.stop_strategy == 'scaling':
                self.correct_num += 1
                break
            
            # 多数投票早停
            if t >= 1:
                fot_ans_lst = [x for x in self.trees_ans if x is not None]
                counter = Counter(fot_ans_lst)
                max_count = max(counter.values()) if counter else 0
                if max_count > self.tree_nums / 2:
                    break
        
        return tree_results

class VolCanoMistralForCausalLM(MistralForCausalLM, VolCanoMetaForCausalLM):
    config_class = VolCanoMistralConfig

    def __init__(self, config):
        super(MistralForCausalLM, self).__init__(config)
        
        self.model = VolCanoMistralModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.init_vision_model()
        
        # 初始化FOT过滤器和蒙特卡洛森林
        self.fot_filter = FOTFilter()
        self.use_fot = True
        self.monte_carlo_forest = None  # 延迟初始化
        
        if hasattr(self, 'vision_generator'):
            zero_img_feature = torch.zeros((1, IMG_TOKEN_NUM, config.hidden_size))
            self.register_buffer('zero_img_feature', zero_img_feature, persistent=False)

        self.post_init()
        self.output_img_id = -100
        self.regression_weight = 1.0
        self.no_bind = False
    
    def initialize_monte_carlo_forest(self, max_iter=2, tree_nums=2, stop_strategy='scaling'):
        """初始化蒙特卡洛森林"""
        self.monte_carlo_forest = MonteCarloForest(
            model=self,
            max_iter=max_iter,
            tree_nums=tree_nums,
            stop_strategy=stop_strategy
        )
    
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
                                     num_thoughts=3, use_monte_carlo=False, **kwargs):
        """
        使用FOT机制的条件完成方法
        支持简单FOT过滤和蒙特卡洛树搜索两种模式
        """
        if not self.use_fot or not hasattr(self, 'fot_filter'):
            return self.condition_completion(input_dict, temperature, max_new_tokens, 
                                           guidance_scale, avoid_image_gen, **kwargs)
        
        # 蒙特卡洛森林模式
        if use_monte_carlo:
            if self.monte_carlo_forest is None:
                self.initialize_monte_carlo_forest()
            
            question = input_dict.get('question', input_dict.get('prompt', ''))
            ground_truth = input_dict.get('ground_truth', None)
            
            tree_results = self.monte_carlo_forest.run_forest(question, ground_truth)
            
            # 选择最佳答案
            best_answer = None
            best_score = -1
            
            for tree_data in tree_results:
                answers = tree_data['answers_list']
                rewards = tree_data['to_explore_reward']
                
                for answer in answers:
                    if answer in rewards and rewards[answer]:
                        avg_reward = np.mean(rewards[answer])
                        if avg_reward > best_score:
                            best_score = avg_reward
                            best_answer = answer
            
            return best_answer if best_answer else "I don't know."
        
        # 简单FOT过滤模式（原有实现）
        else:
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