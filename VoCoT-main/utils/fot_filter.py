import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
import re
from transformers import AutoTokenizer, AutoModel

class FOTFilter:
    """
    Filter of Thoughts (FOT) - 对模型生成的思维进行筛选
    """
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 confidence_threshold=0.7, coherence_threshold=0.6):
        self.confidence_threshold = confidence_threshold
        self.coherence_threshold = coherence_threshold
        
        # 用于计算思维质量的辅助模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def calculate_confidence(self, thought: str, logits: torch.Tensor) -> float:
        """
        计算思维的置信度
        """
        # 基于logits计算置信度
        probs = F.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1)[0]
        confidence = torch.mean(max_probs).item()
        
        # 结合文本长度和复杂度
        text_length_factor = min(len(thought.split()) / 50.0, 1.0)  # 归一化到[0,1]
        
        return confidence * 0.7 + text_length_factor * 0.3
    
    def calculate_coherence(self, thought: str, question: str) -> float:
        """
        计算思维与问题的一致性
        """
        # 使用语义相似度计算一致性
        inputs_thought = self.tokenizer(thought, return_tensors="pt", 
                                       padding=True, truncation=True, max_length=512)
        inputs_question = self.tokenizer(question, return_tensors="pt", 
                                        padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            thought_embedding = self.model(**inputs_thought).last_hidden_state.mean(dim=1)
            question_embedding = self.model(**inputs_question).last_hidden_state.mean(dim=1)
            
            # 计算余弦相似度
            coherence = F.cosine_similarity(thought_embedding, question_embedding, dim=1).item()
        
        return max(0, coherence)  # 确保非负
    
    def check_logical_consistency(self, thought: str) -> float:
        """
        检查思维的逻辑一致性
        """
        # 检查是否包含矛盾表述
        contradiction_patterns = [
            (r'\b(yes|true|correct)\b.*\b(no|false|incorrect)\b', -0.3),
            (r'\b(left)\b.*\b(right)\b.*\b(same)\b', -0.2),
            (r'\b(above)\b.*\b(below)\b.*\b(same)\b', -0.2),
        ]
        
        consistency_score = 1.0
        for pattern, penalty in contradiction_patterns:
            if re.search(pattern, thought.lower()):
                consistency_score += penalty
        
        # 检查是否包含视觉推理关键词
        visual_keywords = ['see', 'look', 'observe', 'visual', 'image', 'picture', 
                          'color', 'shape', 'position', 'location']
        keyword_count = sum(1 for keyword in visual_keywords if keyword in thought.lower())
        keyword_bonus = min(keyword_count * 0.1, 0.3)
        
        return max(0, min(1, consistency_score + keyword_bonus))
    
    def filter_thought(self, thought: str, question: str, logits: torch.Tensor = None) -> Tuple[bool, float]:
        """
        对单个思维进行筛选
        返回: (是否通过筛选, 综合质量分数)
        """
        # 计算各项指标
        confidence = self.calculate_confidence(thought, logits) if logits is not None else 0.8
        coherence = self.calculate_coherence(thought, question)
        consistency = self.check_logical_consistency(thought)
        
        # 综合评分
        overall_score = confidence * 0.4 + coherence * 0.4 + consistency * 0.2
        
        # 判断是否通过筛选
        passed = (confidence >= self.confidence_threshold and 
                 coherence >= self.coherence_threshold and
                 overall_score >= 0.6)
        
        return passed, overall_score
    
    def filter_multiple_thoughts(self, thoughts: List[str], question: str, 
                                logits_list: List[torch.Tensor] = None) -> List[Tuple[str, float]]:
        """
        对多个思维进行筛选和排序
        返回: [(思维, 质量分数)] 按质量分数降序排列
        """
        results = []
        
        for i, thought in enumerate(thoughts):
            logits = logits_list[i] if logits_list else None
            passed, score = self.filter_thought(thought, question, logits)
            
            if passed:
                results.append((thought, score))
        
        # 按质量分数降序排列
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results