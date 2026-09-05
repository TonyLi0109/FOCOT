import torch
import numpy as np
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json
import time

@dataclass
class ThoughtNode:
    """
    思维树节点类
    """
    content: str
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = None
    visits: int = 0
    total_reward: float = 0.0
    depth: int = 0
    confidence: float = 0.0
    coherence: float = 0.0
    consistency: float = 0.0
    is_terminal: bool = False
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    @property
    def average_reward(self) -> float:
        """计算平均奖励"""
        return self.total_reward / max(self.visits, 1)
    
    @property
    def quality_score(self) -> float:
        """计算综合质量分数"""
        return (self.confidence + self.coherence + self.consistency) / 3.0
    
    def add_child(self, child: 'ThoughtNode'):
        """添加子节点"""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
    
    def is_fully_expanded(self, max_children: int = 3) -> bool:
        """检查节点是否已完全扩展"""
        return len(self.children) >= max_children or self.is_terminal
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'content': self.content,
            'visits': self.visits,
            'total_reward': self.total_reward,
            'average_reward': self.average_reward,
            'depth': self.depth,
            'confidence': self.confidence,
            'coherence': self.coherence,
            'consistency': self.consistency,
            'quality_score': self.quality_score,
            'is_terminal': self.is_terminal,
            'num_children': len(self.children)
        }

class UCBCalculator:
    """
    UCB（Upper Confidence Bound）计算器
    """
    
    def __init__(self, c: float = 1.4):
        self.c = c  # 探索常数
    
    def calculate_ucb(self, node: ThoughtNode, parent_visits: int) -> float:
        """
        计算UCB值
        
        Args:
            node: 当前节点
            parent_visits: 父节点访问次数
        
        Returns:
            UCB值
        """
        if node.visits == 0:
            return float('inf')  # 未访问的节点优先级最高
        
        exploitation = node.average_reward
        exploration = self.c * math.sqrt(math.log(parent_visits) / node.visits)
        
        return exploitation + exploration
    
    def select_best_child(self, parent: ThoughtNode) -> Optional[ThoughtNode]:
        """
        根据UCB值选择最佳子节点
        
        Args:
            parent: 父节点
        
        Returns:
            最佳子节点
        """
        if not parent.children:
            return None
        
        best_child = None
        best_ucb = float('-inf')
        
        for child in parent.children:
            ucb = self.calculate_ucb(child, parent.visits)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
        
        return best_child

class ThoughtQualityEvaluator:
    """
    思维质量评估器
    """
    
    def __init__(self, confidence_weight: float = 0.4, 
                 coherence_weight: float = 0.3, 
                 consistency_weight: float = 0.3):
        self.confidence_weight = confidence_weight
        self.coherence_weight = coherence_weight
        self.consistency_weight = consistency_weight
    
    def evaluate_confidence(self, thought: str, context: str = "") -> float:
        """
        评估思维的置信度
        
        Args:
            thought: 思维内容
            context: 上下文
        
        Returns:
            置信度分数 (0-1)
        """
        # 基于长度和结构的简单置信度评估
        if not thought.strip():
            return 0.0
        
        # 长度因子
        length_factor = min(len(thought.split()) / 50.0, 1.0)
        
        # 结构因子（包含推理词汇）
        reasoning_words = ['because', 'therefore', 'since', 'thus', 'hence', 
                          'consequently', 'as a result', 'due to']
        structure_factor = sum(1 for word in reasoning_words if word in thought.lower()) / len(reasoning_words)
        
        # 确定性词汇因子
        certainty_words = ['clearly', 'obviously', 'definitely', 'certainly', 'undoubtedly']
        certainty_factor = min(sum(1 for word in certainty_words if word in thought.lower()) / 3.0, 1.0)
        
        confidence = (length_factor * 0.4 + structure_factor * 0.4 + certainty_factor * 0.2)
        return min(confidence, 1.0)
    
    def evaluate_coherence(self, thought: str, previous_thoughts: List[str] = None) -> float:
        """
        评估思维的连贯性
        
        Args:
            thought: 当前思维
            previous_thoughts: 之前的思维列表
        
        Returns:
            连贯性分数 (0-1)
        """
        if not thought.strip():
            return 0.0
        
        # 基本连贯性：句子结构和逻辑连接词
        sentences = thought.split('.')
        if len(sentences) <= 1:
            return 0.5  # 单句的连贯性中等
        
        # 连接词检查
        connection_words = ['and', 'but', 'however', 'moreover', 'furthermore', 
                           'additionally', 'meanwhile', 'subsequently']
        connection_score = sum(1 for word in connection_words if word in thought.lower()) / len(sentences)
        
        # 与前序思维的一致性
        consistency_score = 0.5
        if previous_thoughts:
            # 简单的词汇重叠检查
            current_words = set(thought.lower().split())
            for prev_thought in previous_thoughts[-3:]:  # 只检查最近3个思维
                prev_words = set(prev_thought.lower().split())
                overlap = len(current_words.intersection(prev_words))
                consistency_score += overlap / max(len(current_words), len(prev_words), 1)
            consistency_score /= len(previous_thoughts[-3:]) + 1
        
        coherence = (connection_score * 0.6 + consistency_score * 0.4)
        return min(coherence, 1.0)
    
    def evaluate_consistency(self, thought: str, question: str, context: str = "") -> float:
        """
        评估思维与问题的一致性
        
        Args:
            thought: 思维内容
            question: 原始问题
            context: 上下文
        
        Returns:
            一致性分数 (0-1)
        """
        if not thought.strip() or not question.strip():
            return 0.0
        
        # 关键词重叠
        thought_words = set(thought.lower().split())
        question_words = set(question.lower().split())
        
        # 移除停用词
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        thought_words -= stop_words
        question_words -= stop_words
        
        if not question_words:
            return 0.5
        
        overlap = len(thought_words.intersection(question_words))
        consistency = overlap / len(question_words)
        
        return min(consistency, 1.0)
    
    def evaluate_thought(self, thought: str, question: str = "", 
                        previous_thoughts: List[str] = None, 
                        context: str = "") -> Dict[str, float]:
        """
        综合评估思维质量
        
        Args:
            thought: 思维内容
            question: 原始问题
            previous_thoughts: 之前的思维列表
            context: 上下文
        
        Returns:
            包含各项评分的字典
        """
        confidence = self.evaluate_confidence(thought, context)
        coherence = self.evaluate_coherence(thought, previous_thoughts)
        consistency = self.evaluate_consistency(thought, question, context)
        
        # 综合分数
        overall_score = (confidence * self.confidence_weight + 
                        coherence * self.coherence_weight + 
                        consistency * self.consistency_weight)
        
        return {
            'confidence': confidence,
            'coherence': coherence,
            'consistency': consistency,
            'overall_score': overall_score
        }

class TreeManager:
    """
    思维树管理器
    """
    
    def __init__(self, max_depth: int = 5, max_children: int = 3):
        self.max_depth = max_depth
        self.max_children = max_children
        self.quality_evaluator = ThoughtQualityEvaluator()
    
    def create_root_node(self, content: str) -> ThoughtNode:
        """创建根节点"""
        return ThoughtNode(content=content, depth=0)
    
    def expand_node(self, node: ThoughtNode, new_thoughts: List[str], 
                   question: str = "") -> List[ThoughtNode]:
        """
        扩展节点
        
        Args:
            node: 要扩展的节点
            new_thoughts: 新思维列表
            question: 原始问题
        
        Returns:
            新创建的子节点列表
        """
        new_children = []
        
        for thought in new_thoughts:
            if len(node.children) >= self.max_children:
                break
            
            child = ThoughtNode(content=thought)
            
            # 评估思维质量
            previous_thoughts = self.get_path_thoughts(node)
            quality_scores = self.quality_evaluator.evaluate_thought(
                thought, question, previous_thoughts
            )
            
            child.confidence = quality_scores['confidence']
            child.coherence = quality_scores['coherence']
            child.consistency = quality_scores['consistency']
            
            node.add_child(child)
            new_children.append(child)
        
        return new_children
    
    def get_path_thoughts(self, node: ThoughtNode) -> List[str]:
        """
        获取从根节点到当前节点的思维路径
        
        Args:
            node: 当前节点
        
        Returns:
            思维路径列表
        """
        path = []
        current = node
        
        while current is not None:
            path.append(current.content)
            current = current.parent
        
        return list(reversed(path))
    
    def get_best_path(self, root: ThoughtNode) -> List[ThoughtNode]:
        """
        获取最佳思维路径
        
        Args:
            root: 根节点
        
        Returns:
            最佳路径节点列表
        """
        path = [root]
        current = root
        
        while current.children:
            # 选择平均奖励最高的子节点
            best_child = max(current.children, key=lambda x: x.average_reward)
            path.append(best_child)
            current = best_child
        
        return path
    
    def get_tree_statistics(self, root: ThoughtNode) -> Dict[str, Any]:
        """
        获取树的统计信息
        
        Args:
            root: 根节点
        
        Returns:
            统计信息字典
        """
        def traverse(node, stats):
            stats['total_nodes'] += 1
            stats['total_visits'] += node.visits
            stats['max_depth'] = max(stats['max_depth'], node.depth)
            
            if node.children:
                stats['internal_nodes'] += 1
                for child in node.children:
                    traverse(child, stats)
            else:
                stats['leaf_nodes'] += 1
        
        stats = {
            'total_nodes': 0,
            'internal_nodes': 0,
            'leaf_nodes': 0,
            'total_visits': 0,
            'max_depth': 0
        }
        
        traverse(root, stats)
        
        # 计算平均分支因子
        if stats['internal_nodes'] > 0:
            stats['avg_branching_factor'] = (stats['total_nodes'] - 1) / stats['internal_nodes']
        else:
            stats['avg_branching_factor'] = 0
        
        return stats

class RewardCalculator:
    """
    奖励计算器
    """
    
    def __init__(self, quality_weight: float = 0.6, 
                 correctness_weight: float = 0.4):
        self.quality_weight = quality_weight
        self.correctness_weight = correctness_weight
    
    def calculate_reward(self, node: ThoughtNode, 
                        is_correct: Optional[bool] = None,
                        ground_truth: str = "") -> float:
        """
        计算节点奖励
        
        Args:
            node: 思维节点
            is_correct: 是否正确（如果已知）
            ground_truth: 标准答案
        
        Returns:
            奖励值 (0-1)
        """
        # 基于质量的奖励
        quality_reward = node.quality_score
        
        # 基于正确性的奖励
        correctness_reward = 0.5  # 默认中性奖励
        if is_correct is not None:
            correctness_reward = 1.0 if is_correct else 0.0
        elif ground_truth and node.content:
            # 简单的文本相似度检查
            node_words = set(node.content.lower().split())
            truth_words = set(ground_truth.lower().split())
            if truth_words:
                overlap = len(node_words.intersection(truth_words))
                correctness_reward = overlap / len(truth_words)
        
        # 综合奖励
        total_reward = (quality_reward * self.quality_weight + 
                       correctness_reward * self.correctness_weight)
        
        return min(max(total_reward, 0.0), 1.0)
    
    def update_node_reward(self, node: ThoughtNode, reward: float):
        """
        更新节点奖励
        
        Args:
            node: 要更新的节点
            reward: 新的奖励值
        """
        node.visits += 1
        node.total_reward += reward
    
    def backpropagate_reward(self, leaf_node: ThoughtNode, reward: float):
        """
        向上传播奖励
        
        Args:
            leaf_node: 叶子节点
            reward: 奖励值
        """
        current = leaf_node
        while current is not None:
            self.update_node_reward(current, reward)
            current = current.parent

def save_tree_to_json(root: ThoughtNode, filepath: str):
    """
    将思维树保存为JSON文件
    
    Args:
        root: 根节点
        filepath: 保存路径
    """
    def node_to_dict(node):
        result = node.to_dict()
        result['children'] = [node_to_dict(child) for child in node.children]
        return result
    
    tree_data = {
        'tree': node_to_dict(root),
        'timestamp': time.time(),
        'statistics': TreeManager().get_tree_statistics(root)
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(tree_data, f, indent=2, ensure_ascii=False)

def load_tree_from_json(filepath: str) -> ThoughtNode:
    """
    从JSON文件加载思维树
    
    Args:
        filepath: 文件路径
    
    Returns:
        根节点
    """
    def dict_to_node(data, parent=None):
        node = ThoughtNode(
            content=data['content'],
            parent=parent,
            visits=data['visits'],
            total_reward=data['total_reward'],
            depth=data['depth'],
            confidence=data['confidence'],
            coherence=data['coherence'],
            consistency=data['consistency'],
            is_terminal=data['is_terminal']
        )
        
        for child_data in data.get('children', []):
            child = dict_to_node(child_data, node)
            node.children.append(child)
        
        return node
    
    with open(filepath, 'r', encoding='utf-8') as f:
        tree_data = json.load(f)
    
    return dict_to_node(tree_data['tree'])

# 导出主要类和函数
__all__ = [
    'ThoughtNode',
    'UCBCalculator', 
    'ThoughtQualityEvaluator',
    'TreeManager',
    'RewardCalculator',
    'save_tree_to_json',
    'load_tree_from_json'
]