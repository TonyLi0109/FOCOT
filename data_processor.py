import json
import re
from typing import Dict, List, Tuple

class OutputProcessor:
    def __init__(self):
        pass
    
    def parse_pope_fot_output(self, output_file: str) -> List[Dict]:
        """
        解析POPE FOT方法的输出文件
        """
        results = []
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 如果是JSON格式的结果
            if isinstance(data, list):
                for item in data:
                    result = {
                        'question_id': item.get('question_id', ''),
                        'question': item.get('question', ''),
                        'prediction': self.extract_yes_no_answer(item.get('fot_prediction', '')),
                        'ground_truth': item.get('label', item.get('ground_truth', '')),
                        'image_path': item.get('image_path', ''),
                        'thoughts': self.extract_fot_thoughts(item.get('fot_reasoning', ''))
                    }
                    results.append(result)
            else:
                # 处理单个结果的情况
                result = {
                    'question_id': data.get('question_id', ''),
                    'question': data.get('question', ''),
                    'prediction': self.extract_yes_no_answer(data.get('fot_prediction', '')),
                    'ground_truth': data.get('label', data.get('ground_truth', '')),
                    'image_path': data.get('image_path', ''),
                    'thoughts': self.extract_fot_thoughts(data.get('fot_reasoning', ''))
                }
                results.append(result)
                
        except json.JSONDecodeError:
            # 如果不是JSON格式，尝试解析文本格式
            results = self.parse_pope_fot_text_output(output_file)
        
        print(f"解析FOT输出: {len(results)} 个结果")
        return results
    
    def parse_amber_fot_output(self, output_file: str) -> List[Dict]:
        """
        解析AMBER FOT方法的输出文件（fot_amber_results.json）
        提取prediction和thoughts字段
        """
        results = []
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 如果是JSON格式的结果
            if isinstance(data, list):
                for item in data:
                    # 提取thoughts中的thought内容
                    thoughts = []
                    if 'thoughts' in item and isinstance(item['thoughts'], list):
                        for thought_item in item['thoughts']:
                            if isinstance(thought_item, dict) and 'thought' in thought_item:
                                thoughts.append(thought_item['thought'])
                    
                    result = {
                        'item_id': item.get('item_id', ''),
                        'question': item.get('question', ''),
                        'prediction': item.get('prediction', ''),
                        'ground_truth': item.get('label', item.get('ground_truth', '')),
                        'image_path': item.get('image_path', ''),
                        'thoughts': thoughts,
                        'num_thoughts': len(thoughts)
                    }
                    results.append(result)
            else:
                # 处理单个结果的情况
                thoughts = []
                if 'thoughts' in data and isinstance(data['thoughts'], list):
                    for thought_item in data['thoughts']:
                        if isinstance(thought_item, dict) and 'thought' in thought_item:
                            thoughts.append(thought_item['thought'])
                
                result = {
                    'item_id': data.get('item_id', ''),
                    'question': data.get('question', ''),
                    'prediction': data.get('prediction', ''),
                    'ground_truth': data.get('label', data.get('ground_truth', '')),
                    'image_path': data.get('image_path', ''),
                    'thoughts': thoughts,
                    'num_thoughts': len(thoughts)
                }
                results.append(result)
                
        except json.JSONDecodeError:
            # 如果不是JSON格式，尝试解析文本格式
            results = self.parse_amber_fot_text_output(output_file)
        
        print(f"解析AMBER FOT输出: {len(results)} 个结果")
        return results
    
    def parse_amber_fot_text_output(self, output_file: str) -> List[Dict]:
        """
        解析AMBER FOT方法的文本格式输出文件
        """
        results = []
        
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按条目分割
        entries = content.split('\n\n')
        
        for entry in entries:
            if not entry.strip():
                continue
            
            # 提取各字段
            item_id = self.extract_field_from_text(entry, r'Item ID:\s*(.+)')
            question = self.extract_field_from_text(entry, r'Question:\s*(.+)')
            prediction = self.extract_field_from_text(entry, r'Prediction:\s*(.+)')
            ground_truth = self.extract_field_from_text(entry, r'Ground Truth:\s*(.+)')
            
            # 提取思考过程
            thoughts = self.extract_amber_fot_thoughts(entry)
            
            result = {
                'item_id': item_id,
                'question': question,
                'prediction': prediction,
                'ground_truth': ground_truth,
                'thoughts': thoughts,
                'num_thoughts': len(thoughts)
            }
            results.append(result)
        
        return results
    
    def parse_amber_cot_output(self, output_file: str) -> List[Dict]:
        """
        解析AMBER VoCoT方法的输出文件（AMBER_desc.json）
        提取prediction作为推理过程
        """
        results = []
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 如果是JSON格式的结果
            if isinstance(data, list):
                for item in data:
                    result = {
                        'item_id': item.get('item_id', ''),
                        'question': item.get('question', ''),
                        'prediction': item.get('prediction', ''),
                        'ground_truth': item.get('label', item.get('ground_truth', '')),
                        'image_path': item.get('image_path', ''),
                        'reasoning': item.get('prediction', '')  # VoCoT的prediction就是推理过程
                    }
                    results.append(result)
            else:
                # 处理单个结果的情况
                result = {
                    'item_id': data.get('item_id', ''),
                    'question': data.get('question', ''),
                    'prediction': data.get('prediction', ''),
                    'ground_truth': data.get('label', data.get('ground_truth', '')),
                    'image_path': data.get('image_path', ''),
                    'reasoning': data.get('prediction', '')
                }
                results.append(result)
                
        except json.JSONDecodeError:
            # 如果不是JSON格式，尝试解析文本格式
            results = self.parse_amber_cot_text_output(output_file)
        
        print(f"解析AMBER VoCoT输出: {len(results)} 个结果")
        return results
    
    def parse_amber_cot_text_output(self, output_file: str) -> List[Dict]:
        """
        解析AMBER COT方法的文本格式输出文件
        """
        results = []
        
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按条目分割
        entries = content.split('\n\n')
        
        for entry in entries:
            if not entry.strip():
                continue
            
            # 提取各字段
            item_id = self.extract_field_from_text(entry, r'Item ID:\s*(.+)')
            question = self.extract_field_from_text(entry, r'Question:\s*(.+)')
            prediction = self.extract_field_from_text(entry, r'Prediction:\s*(.+)')
            ground_truth = self.extract_field_from_text(entry, r'Ground Truth:\s*(.+)')
            reasoning = self.extract_field_from_text(entry, r'Reasoning:\s*(.+)', multiline=True)
            
            result = {
                'item_id': item_id,
                'question': question,
                'prediction': prediction,
                'ground_truth': ground_truth,
                'reasoning': reasoning
            }
            results.append(result)
        
        return results
    
    def parse_pope_fot_text_output(self, output_file: str) -> List[Dict]:
        """
        解析POPE FOT方法的文本格式输出文件
        """
        results = []
        
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按条目分割
        entries = content.split('\n\n')
        
        for entry in entries:
            if not entry.strip():
                continue
            
            # 提取各字段
            question_id = self.extract_field_from_text(entry, r'Question ID:\s*(.+)')
            question = self.extract_field_from_text(entry, r'Question:\s*(.+)')
            prediction = self.extract_field_from_text(entry, r'Prediction:\s*(.+)')
            ground_truth = self.extract_field_from_text(entry, r'Ground Truth:\s*(.+)')
            
            # 提取思考过程
            thoughts = self.extract_fot_thoughts(entry)
            
            result = {
                'question_id': question_id,
                'question': question,
                'prediction': self.extract_yes_no_answer(prediction),
                'ground_truth': ground_truth,
                'thoughts': thoughts
            }
            results.append(result)
        
        return results
    
    def parse_pope_cot_output(self, output_file: str) -> List[Dict]:
        """
        解析POPE COT方法的输出文件
        """
        results = []
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 如果是JSON格式的结果
            if isinstance(data, list):
                for item in data:
                    result = {
                        'question_id': item.get('question_id', ''),
                        'question': item.get('question', ''),
                        'prediction': self.extract_yes_no_answer(item.get('cot_prediction', '')),
                        'ground_truth': item.get('label', item.get('ground_truth', '')),
                        'image_path': item.get('image_path', ''),
                        'reasoning': item.get('cot_reasoning', '')
                    }
                    results.append(result)
            else:
                # 处理单个结果的情况
                result = {
                    'question_id': data.get('question_id', ''),
                    'question': data.get('question', ''),
                    'prediction': self.extract_yes_no_answer(data.get('cot_prediction', '')),
                    'ground_truth': data.get('label', data.get('ground_truth', '')),
                    'image_path': data.get('image_path', ''),
                    'reasoning': data.get('cot_reasoning', '')
                }
                results.append(result)
                
        except json.JSONDecodeError:
            # 如果不是JSON格式，尝试解析文本格式
            results = self.parse_pope_cot_text_output(output_file)
        
        print(f"解析COT输出: {len(results)} 个结果")
        return results
    
    def parse_pope_cot_text_output(self, output_file: str) -> List[Dict]:
        """
        解析POPE COT方法的文本格式输出文件
        """
        results = []
        
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按条目分割
        entries = content.split('\n\n')
        
        for entry in entries:
            if not entry.strip():
                continue
            
            # 提取各字段
            question_id = self.extract_field_from_text(entry, r'Question ID:\s*(.+)')
            question = self.extract_field_from_text(entry, r'Question:\s*(.+)')
            prediction = self.extract_field_from_text(entry, r'Prediction:\s*(.+)')
            ground_truth = self.extract_field_from_text(entry, r'Ground Truth:\s*(.+)')
            reasoning = self.extract_field_from_text(entry, r'Reasoning:\s*(.+)', multiline=True)
            
            result = {
                'question_id': question_id,
                'question': question,
                'prediction': self.extract_yes_no_answer(prediction),
                'ground_truth': ground_truth,
                'reasoning': reasoning
            }
            results.append(result)
        
        return results
    
    def extract_fot_thoughts(self, reasoning_text: str) -> List[str]:
        """
        从FOT推理文本中提取思考步骤
        """
        thoughts = []
        
        if not reasoning_text:
            return thoughts
        
        # 匹配不同格式的思考步骤
        patterns = [
            r'Step (\d+):\s*(.+?)(?=Step \d+:|$)',
            r'思考步骤 (\d+):\s*(.+?)(?=思考步骤 \d+:|$)',
            r'Thought (\d+):\s*(.+?)(?=Thought \d+:|$)',
            r'候选答案 (\d+):\s*(.+?)(?=候选答案 \d+:|$)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, reasoning_text, re.DOTALL | re.IGNORECASE)
            if matches:
                for step_num, content in matches:
                    thoughts.append(f"Step {step_num}: {content.strip()}")
                break
        
        # 如果没有匹配到结构化的步骤，尝试按行分割
        if not thoughts:
            lines = reasoning_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:  # 过滤掉太短的行
                    thoughts.append(line)
        
        return thoughts
    
    def extract_amber_fot_thoughts(self, reasoning_text: str) -> List[str]:
        """
        从AMBER FOT推理文本中提取思考步骤
        """
        thoughts = []
        
        if not reasoning_text:
            return thoughts
        
        # 匹配AMBER特定的思考步骤格式
        patterns = [
            r'Thought (\d+):\s*(.+?)(?=Thought \d+:|$)',
            r'Step (\d+):\s*(.+?)(?=Step \d+:|$)',
            r'思考 (\d+):\s*(.+?)(?=思考 \d+:|$)',
            r'推理步骤 (\d+):\s*(.+?)(?=推理步骤 \d+:|$)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, reasoning_text, re.DOTALL | re.IGNORECASE)
            if matches:
                for step_num, content in matches:
                    thoughts.append(f"Thought {step_num}: {content.strip()}")
                break
        
        # 如果没有匹配到结构化的步骤，尝试按行分割
        if not thoughts:
            lines = reasoning_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:  # 过滤掉太短的行
                    thoughts.append(line)
        
        return thoughts

    def extract_yes_no_answer(self, text: str) -> str:
        """
        从文本中提取yes/no答案
        """
        if not text:
            return "no"
        
        text = text.lower().strip()
        
        # 直接匹配
        if text in ['yes', 'no']:
            return text
        
        # 开头匹配
        if text.startswith('yes'):
            return 'yes'
        elif text.startswith('no'):
            return 'no'
        
        # 包含匹配
        if 'yes' in text and 'no' not in text:
            return 'yes'
        elif 'no' in text and 'yes' not in text:
            return 'no'
        
        # 默认返回no（保守策略）
        return 'no'
    
    def extract_field_from_text(self, text: str, pattern: str, multiline: bool = False) -> str:
        """
        从文本中提取指定字段
        """
        flags = re.DOTALL if multiline else 0
        match = re.search(pattern, text, flags)
        return match.group(1).strip() if match else ""
    
    # 保留原有方法以兼容其他任务
    def parse_baseline_output(self, output_file: str) -> List[Dict]:
        """
        解析baseline方法的输出文件
        """
        results = []
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for item in data:
            results.append({
                'question_id': item.get('questionId', ''),
                'question': item.get('question', ''),
                'prediction': item.get('prediction', ''),
                'ground_truth': item.get('answer', '')
            })
        return results
    
    def parse_fot_output(self, output_file: str) -> List[Dict]:
        """
        解析FOT方法的输出文件，提取思考过程
        """
        results = []
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 解析FOT输出中的思考步骤
        entries = content.split('\n\n')  # 假设每个问题用双换行分隔
        
        for entry in entries:
            if not entry.strip():
                continue
                
            # 提取问题ID、问题、答案和思考过程
            question_id = self.extract_field(entry, 'Question ID')
            question = self.extract_field(entry, 'Question')
            prediction = self.extract_field(entry, 'Generated final answer')
            ground_truth = self.extract_field(entry, 'Ground truth')
            
            # 提取思考步骤
            thoughts = self.extract_thoughts(entry)
            
            results.append({
                'question_id': question_id,
                'question': question,
                'prediction': prediction,
                'ground_truth': ground_truth,
                'thoughts': thoughts
            })
            
        return results
    
    def extract_field(self, text: str, field_name: str) -> str:
        """
        从文本中提取指定字段
        """
        pattern = f"{field_name}:\s*(.+?)(?=\n[A-Z]|$)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def extract_thoughts(self, text: str) -> List[str]:
        """
        提取FOT思考过程
        """
        thoughts = []
        
        # 提取各个步骤的思考
        step_patterns = [
            r"Step \d+ - candidate \d+ (.+?):\s*(.+?)(?=Step|Generated|$)",
            r"Phase \d+:\s*(.+?)(?=Phase|Generated|$)",
            r"候选答案 \d+:\s*(.+?)(?=候选答案|Generated|$)"
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    thoughts.append(f"{match[0]}: {match[1].strip()}")
                else:
                    thoughts.append(match.strip())
                    
        return thoughts