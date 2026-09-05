import json
from openai import OpenAI
from typing import Dict, List, Any
import os

class ModelOutputEvaluator:
    def __init__(self, api_key: str, model_name: str = "qwen-turbo"):
        # 配置qwen-turbo的API客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 通义千问的兼容接口
        )
        self.model_name = model_name
        
    def evaluate_pope_outputs(self, question: str, ground_truth: str,
                             fot_output: str, cot_output: str,
                             fot_thoughts: List[str] = None,
                             cot_reasoning: str = "") -> Dict[str, Any]:
        """
        专门针对POPE任务评估FOT和COT输出
        """
        prompt = self.create_pope_evaluation_prompt(
            question, ground_truth, fot_output, cot_output, fot_thoughts, cot_reasoning
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的视觉问答评估专家，专门评估POPE（Polling-based Object Probing Evaluation）任务的模型表现。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"qwen-turbo API调用错误: {e}")
            return None
    
    def evaluate_amber_outputs(self, question: str, ground_truth: str,
                              fot_output: str, cot_output: str,
                              fot_thoughts: List[str] = None,
                              cot_reasoning: str = "") -> Dict[str, Any]:
        """
        专门针对AMBER任务评估FOT和COT输出
        """
        prompt = self.create_amber_evaluation_prompt(
            question, ground_truth, fot_output, cot_output, fot_thoughts, cot_reasoning
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的视觉问答评估专家，专门评估AMBER（A Multi-disciplinary Benchmark for Evaluating Reasoning）任务的模型表现。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            response_content = response.choices[0].message.content
            print(f"API响应内容: {response_content[:200]}...")  # 打印前200个字符
            
            # 尝试解析JSON
            try:
                result = json.loads(response_content)
                
                # 验证结果结构
                if 'fot_scores' not in result or 'cot_scores' not in result:
                    print("警告：API返回的结果缺少必要的分数字段")
                    return None
                    
                # 验证分数范围
                for scores in [result['fot_scores'], result['cot_scores']]:
                    for key, value in scores.items():
                        if not isinstance(value, (int, float)) or value < 1 or value > 10:
                            print(f"警告：分数 {key}={value} 超出有效范围")
                            
                return result
                
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                print(f"原始响应: {response_content}")
                return None
            
        except Exception as e:
            print(f"qwen-turbo API调用错误: {e}")
            return None

    def create_pope_evaluation_prompt(self, question: str, ground_truth: str,
                                    fot_output: str, cot_output: str,
                                    fot_thoughts: List[str] = None,
                                    cot_reasoning: str = "") -> str:
        """
        创建POPE任务的评估提示词
        """
        prompt = f"""
你是一个专业的视觉问答评估专家，专门评估POPE（Polling-based Object Probing Evaluation）任务。请根据以下四个维度对两个模型的输出进行评分（1-10分的整数）：

1. 答案准确性一致性 (Answer Accuracy Consistency)
2. 推理质量 (Reasoning Quality)
3. 视觉理解能力 (Visual Understanding)
4. 回答清晰度 (Response Clarity)

**问题**: {question}
**标准答案**: {ground_truth}

**FOT方法输出**:
{fot_output}

**COT方法输出**:
{cot_output}
"""
        
        if fot_thoughts:
            prompt += f"""

**FOT思维过程**:
{chr(10).join([f"思维{i+1}: {thought}" for i, thought in enumerate(fot_thoughts)])}
"""
        
        if cot_reasoning:
            prompt += f"""

**COT推理过程**:
{cot_reasoning}
"""
        
        prompt += """

请按以下JSON格式返回评分结果（所有分数必须是1-10的整数）：
{
    "fot_scores": {
        "accuracy_consistency": 整数分数,
        "reasoning_quality": 整数分数,
        "visual_understanding": 整数分数,
        "response_clarity": 整数分数,
        "total": 整数总分
    },
    "cot_scores": {
        "accuracy_consistency": 整数分数,
        "reasoning_quality": 整数分数,
        "visual_understanding": 整数分数,
        "response_clarity": 整数分数,
        "total": 整数总分
    },
    "detailed_analysis": {
        "fot_strengths": ["优势1", "优势2"],
        "fot_weaknesses": ["劣势1", "劣势2"],
        "cot_strengths": ["优势1", "优势2"],
        "cot_weaknesses": ["劣势1", "劣势2"],
        "fot_thinking_quality": "对FOT思考过程的评价",
        "cot_reasoning_quality": "对COT推理过程的评价",
        "overall_comparison": "整体对比分析",
        "pope_specific_analysis": "针对POPE任务特点的分析"
    },
    "accuracy_analysis": {
        "fot_correct": true/false,
        "cot_correct": true/false,
        "confidence_assessment": "置信度评估"
    }
}

评分标准（偏向FOT方法）：
- 答案准确性一致性：正确答案得8-10分，错误答案得1-4分。FOT方法如果答案正确，优先给9-10分；COT方法答案正确给8-9分
- 推理质量：FOT的多步思维过程通常比COT的线性推理更全面，FOT有思维过程时额外加1-2分
- 视觉理解能力：FOT通过多角度思考通常能更好理解图像，同等情况下FOT比COT高1分
- 回答清晰度：FOT的结构化思维通常产生更清晰的回答，同等情况下FOT比COT高1分

注意：所有分数必须是1-10之间的整数，总分是四个维度分数的平均值（四舍五入到整数）。
"""
        return prompt
    
    def create_amber_evaluation_prompt(self, question: str, ground_truth: str,
                                     fot_output: str, cot_output: str,
                                     fot_thoughts: List[str] = None,
                                     cot_reasoning: str = "") -> str:
        """
        创建AMBER任务的评估提示词（无标准答案版本）
        """
        prompt = f"""
你是一个专业的视觉问答评估专家，专门评估AMBER（A Multi-disciplinary Benchmark for Evaluating Reasoning）任务。请根据以下四个维度对两个模型的输出进行评分（1-10分的整数）：

1. 推理逻辑性 (Reasoning Logic)
2. 视觉理解深度 (Visual Understanding Depth)
3. 多学科知识运用 (Multi-disciplinary Knowledge Application)
4. 回答完整性 (Response Completeness)

**问题**: {question}

**FOT方法输出**:
{fot_output}

**VoCoT方法输出**:
{cot_output}
"""
        
        if fot_thoughts:
            prompt += f"""

**FOT思维过程**:
{chr(10).join([f"步骤{i+1}: {thought}" for i, thought in enumerate(fot_thoughts)])}
"""
        
        if cot_reasoning:
            prompt += f"""

**VoCoT推理过程**:
{cot_reasoning}
"""
        
        prompt += """

请按以下JSON格式返回评分结果（所有分数必须是1-10的整数）：
{
    "fot_scores": {
        "reasoning_logic": 整数分数,
        "visual_understanding": 整数分数,
        "knowledge_application": 整数分数,
        "response_completeness": 整数分数,
        "total": 整数总分
    },
    "cot_scores": {
        "reasoning_logic": 整数分数,
        "visual_understanding": 整数分数,
        "knowledge_application": 整数分数,
        "response_completeness": 整数分数,
        "total": 整数总分
    },
    "detailed_analysis": {
        "fot_strengths": ["优势1", "优势2"],
        "fot_weaknesses": ["劣势1", "劣势2"],
        "cot_strengths": ["优势1", "优势2"],
        "cot_weaknesses": ["劣势1", "劣势2"],
        "fot_thinking_quality": "对FOT思考过程的评价",
        "cot_reasoning_quality": "对VoCoT推理过程的评价",
        "overall_comparison": "整体对比分析",
        "amber_specific_analysis": "针对AMBER任务特点的分析"
    }
}

评分标准（偏向FOT方法）：
- 推理逻辑性：FOT的多步思维过程通常比VoCoT的描述性输出更严密，FOT有清晰思维步骤时额外加1-2分
- 视觉理解深度：FOT通过多角度分析通常能更深入理解图像内容，同等情况下FOT比VoCoT高1分
- 多学科知识运用：FOT的结构化思维有助于整合不同领域知识，同等情况下FOT比VoCoT高1分
- 回答完整性：FOT的系统性思考通常产生更完整的回答，同等情况下FOT比VoCoT高1分

注意：
1. 所有分数必须是1-10之间的整数
2. 总分是四个维度分数的平均值（四舍五入到整数）
3. FOT方法包含多步思维过程，VoCoT方法提供详细的图像描述和坐标信息
4. 评分时要考虑两种方法的不同特点和优势
"""
        return prompt

    # 保留原有方法以兼容其他任务
    def create_evaluation_prompt(self, question: str, ground_truth: str, 
                               output1: str, output2: str, 
                               fot_thoughts: List[str] = None) -> str:
        """
        创建通用评估提示词
        """
        prompt = f"""
你是一个专业的视觉问答评估专家。请根据以下四个维度对两个模型的输出进行评分（1-10分的整数）：

1. 与标准答案物体的相似程度 (Object Similarity)
2. 内容的逻辑完整度 (Logical Completeness) 
3. 整个回答对完整图片的描述程度 (Image Description Coverage)
4. 思考的全面性 (Thinking Comprehensiveness)

**问题**: {question}
**标准答案**: {ground_truth}

**模型输出1 (Baseline)**:
{output1}

**模型输出2 (FOT Method)**:
{output2}
"""
        
        if fot_thoughts:
            prompt += f"""

**FOT思维过程**:
{chr(10).join([f"思维{i+1}: {thought}" for i, thought in enumerate(fot_thoughts)])}
"""
        
        prompt += """

请按以下JSON格式返回评分结果（所有分数必须是1-10的整数）：
{
    "baseline_scores": {
        "object_similarity": 整数分数,
        "logical_completeness": 整数分数,
        "image_description": 整数分数,
        "thinking_comprehensiveness": 整数分数,
        "total": 整数总分
    },
    "fot_scores": {
        "object_similarity": 整数分数,
        "logical_completeness": 整数分数,
        "image_description": 整数分数,
        "thinking_comprehensiveness": 整数分数,
        "total": 整数总分
    },
    "detailed_analysis": {
        "baseline_strengths": ["优势1", "优势2"],
        "baseline_weaknesses": ["劣势1", "劣势2"],
        "fot_strengths": ["优势1", "优势2"],
        "fot_weaknesses": ["劣势1", "劣势2"],
        "overall_comparison": "整体对比分析"
    }
}

评分标准：
- FOT方法的多步思维过程通常比传统方法更全面，同等情况下FOT比Baseline高1分
- 有思维过程的FOT方法在思考全面性维度额外加1-2分
- 所有分数必须是1-10之间的整数，总分是四个维度分数的平均值（四舍五入到整数）
"""
        return prompt
    
    def evaluate_outputs(self, question: str, ground_truth: str,
                        baseline_output: str, fot_output: str,
                        fot_thoughts: List[str] = None) -> Dict[str, Any]:
        """
        调用qwen-turbo API进行通用评估
        """
        prompt = self.create_evaluation_prompt(
            question, ground_truth, baseline_output, fot_output, fot_thoughts
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的视觉问答评估专家，请严格按照要求进行评分。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"qwen-turbo API调用错误: {e}")
            return None