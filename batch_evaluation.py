import json
import os
from typing import List, Dict
from eval_comparison_system import ModelOutputEvaluator
from data_processor import OutputProcessor
import pandas as pd

class BatchEvaluator:
    def __init__(self, api_key: str, model_name: str = "qwen-turbo"):
        self.evaluator = ModelOutputEvaluator(api_key, model_name)
        self.processor = OutputProcessor()
    
    def run_pope_evaluation(self, fot_file: str, cot_file: str, 
                           output_file: str = "pope_evaluation_results.json"):
        """
        运行POPE数据集的FOT vs COT批量评估
        """
        print("开始POPE评估...")
        
        # 解析输出文件
        fot_results = self.processor.parse_pope_fot_output(fot_file)
        cot_results = self.processor.parse_pope_cot_output(cot_file)
        
        # 匹配相同问题的结果
        matched_pairs = self.match_pope_results(fot_results, cot_results)
        
        if not matched_pairs:
            print("没有找到匹配的问题对")
            return []
        
        evaluation_results = []
        
        # 逐个评估
        for i, (fot_item, cot_item) in enumerate(matched_pairs):
            print(f"评估进度: {i+1}/{len(matched_pairs)}")
            
            # 调用评估API
            eval_result = self.evaluator.evaluate_pope_outputs(
                question=fot_item['question'],
                ground_truth=fot_item['ground_truth'],
                fot_output=fot_item['prediction'],
                cot_output=cot_item['prediction'],
                fot_thoughts=fot_item.get('thoughts', []),
                cot_reasoning=cot_item.get('reasoning', '')
            )
            
            if eval_result:
                # 整合结果
                result = {
                    'question_id': fot_item['question_id'],
                    'question': fot_item['question'],
                    'ground_truth': fot_item['ground_truth'],
                    'fot_prediction': fot_item['prediction'],
                    'cot_prediction': cot_item['prediction'],
                    'fot_scores': eval_result['fot_scores'],
                    'cot_scores': eval_result['cot_scores'],
                    'detailed_analysis': eval_result['detailed_analysis'],
                    'accuracy_analysis': eval_result['accuracy_analysis']
                }
                evaluation_results.append(result)
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
            
        # 生成统计报告
        self.generate_pope_report(evaluation_results)
        
        return evaluation_results
    
    def run_amber_evaluation(self, fot_file: str, cot_file: str, 
                            output_file: str = "amber_evaluation_results.json"):
        """
        运行AMBER数据集的FOT vs COT批量评估
        注意：fot_file应该是fot_amber_results.json（包含thoughts的FOT输出）
              cot_file应该是AMBER_desc.json（VoCoT的详细描述输出）
        """
        print("开始AMBER评估...")
        print(f"FOT文件: {fot_file}")
        print(f"COT文件: {cot_file}")
        
        # 解析输出文件 - 修正文件对应关系
        fot_results = self.processor.parse_amber_fot_output(fot_file)  # fot_amber_results.json
        cot_results = self.processor.parse_amber_cot_output(cot_file)  # AMBER_desc.json
        
        print(f"FOT结果数量: {len(fot_results)}")
        print(f"COT结果数量: {len(cot_results)}")
        
        # 打印前几个结果的格式确认
        if fot_results:
            print(f"FOT结果示例: {fot_results[0]}")
        if cot_results:
            print(f"COT结果示例: {cot_results[0]}")
        
        # 匹配相同问题的结果
        matched_pairs = self.match_amber_results(fot_results, cot_results)
        
        if not matched_pairs:
            print("没有找到匹配的问题对")
            return []

        print(f"匹配的问题对数量: {len(matched_pairs)}")

        evaluation_results = []
        failed_evaluations = 0

        # 逐个评估
        for i, (fot_item, cot_item) in enumerate(matched_pairs):
            print(f"评估进度: {i+1}/{len(matched_pairs)}")
            
            # 打印调试信息
            if i < 3:  # 只打印前3个的详细信息
                print(f"  Item ID: {fot_item.get('item_id', '')}")
                print(f"  标准答案: {fot_item['ground_truth']}")
                print(f"  FOT预测: {fot_item['prediction'][:100]}...")
                print(f"  FOT思维数量: {len(fot_item.get('thoughts', []))}")
                print(f"  COT预测: {cot_item['prediction'][:100]}...")
            
            # 调用评估API
            eval_result = self.evaluator.evaluate_amber_outputs(
                question=fot_item.get('question', ''),
                ground_truth=str(fot_item['ground_truth']),
                fot_output=fot_item['prediction'],
                cot_output=cot_item['prediction'],
                fot_thoughts=fot_item.get('thoughts', []),
                cot_reasoning=cot_item.get('reasoning', '')
            )
            
            if eval_result:
                # 打印评估结果的调试信息
                if i < 3:
                    print(f"  评估结果: {eval_result}")
                
                # 整合结果
                result = {
                    'item_id': fot_item['item_id'],
                    'question': fot_item.get('question', ''),
                    'ground_truth': fot_item['ground_truth'],
                    'fot_output': fot_item['prediction'],
                    'cot_output': cot_item['prediction'],
                    'fot_thoughts': fot_item.get('thoughts', []),
                    'cot_reasoning': cot_item.get('reasoning', ''),
                    'evaluation': eval_result
                }
                evaluation_results.append(result)
            else:
                failed_evaluations += 1
                print(f"  评估失败，使用默认分数")
                # 使用默认分数
                default_scores = {
                    "物体相似度": 5.0,
                    "逻辑完整度": 5.0, 
                    "图片描述程度": 5.0,
                    "思考全面性": 5.0,
                    "分析": "API调用失败，使用默认分数"
                }
                
                result = {
                    'item_id': fot_item['item_id'],
                    'question': fot_item.get('question', ''),
                    'ground_truth': fot_item['ground_truth'],
                    'fot_output': fot_item['prediction'],
                    'cot_output': cot_item['prediction'],
                    'fot_thoughts': fot_item.get('thoughts', []),
                    'cot_reasoning': cot_item.get('reasoning', ''),
                    'evaluation': default_scores
                }
                evaluation_results.append(result)

        print(f"评估完成，成功: {len(evaluation_results) - failed_evaluations}, 失败: {failed_evaluations}")

        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

        # 生成报告
        # 生成报告
        report_file = output_file.replace('.json', '_report.json')
        self.generate_amber_report(evaluation_results, report_file)

        return evaluation_results
        
    def match_pope_results(self, fot: List[Dict], cot: List[Dict]) -> List[tuple]:
        """
        匹配相同问题的POPE结果
        """
        fot_dict = {item['question_id']: item for item in fot}
        matched_pairs = []
        
        for cot_item in cot:
            qid = cot_item['question_id']
            if qid in fot_dict:
                matched_pairs.append((fot_dict[qid], cot_item))
                
        print(f"成功匹配 {len(matched_pairs)} 个问题对")
        return matched_pairs
    
    def match_amber_results(self, fot: List[Dict], cot: List[Dict]) -> List[tuple]:
        """
        匹配相同问题的AMBER结果
        """
        fot_dict = {item['item_id']: item for item in fot}
        matched_pairs = []
        
        for cot_item in cot:
            item_id = cot_item['item_id']
            if item_id in fot_dict:
                matched_pairs.append((fot_dict[item_id], cot_item))
                
        print(f"成功匹配 {len(matched_pairs)} 个问题对")
        return matched_pairs

    def generate_pope_report(self, results: List[Dict]):
        """
        生成POPE评估报告
        """
        if not results:
            print("没有评估结果")
            return
            
        # 计算准确率 - 修复ground_truth类型问题
        fot_correct = sum(1 for r in results if str(r['fot_prediction']).lower() == str(r['ground_truth']).lower())
        cot_correct = sum(1 for r in results if str(r['cot_prediction']).lower() == str(r['ground_truth']).lower())
        total = len(results)
        
        fot_accuracy = fot_correct / total * 100
        cot_accuracy = cot_correct / total * 100
        
        # 计算平均分
        fot_scores = []
        cot_scores = []
        
        for result in results:
            fot_scores.append(result['fot_scores'])
            cot_scores.append(result['cot_scores'])
        
        # 计算各维度平均分
        dimensions = ['accuracy_consistency', 'reasoning_quality', 
                     'visual_understanding', 'response_clarity', 'total']
        
        print("\n=== POPE评估报告 ===")
        print(f"总评估样本数: {total}")
        print(f"\n准确率对比:")
        print(f"FOT准确率: {fot_accuracy:.2f}% ({fot_correct}/{total})")
        print(f"COT准确率: {cot_accuracy:.2f}% ({cot_correct}/{total})")
        print(f"准确率差值: {fot_accuracy - cot_accuracy:+.2f}%")
        
        print("\n各维度平均分对比:")
        
        for dim in dimensions:
            fot_avg = sum(score[dim] for score in fot_scores) / len(fot_scores)
            cot_avg = sum(score[dim] for score in cot_scores) / len(cot_scores)
            
            print(f"{dim}:")
            print(f"  FOT: {fot_avg:.2f}")
            print(f"  COT: {cot_avg:.2f}")
            print(f"  差值: {fot_avg - cot_avg:+.2f}")
            print()
        
        # 保存详细报告
        df_data = []
        dimensions = ['accuracy_consistency', 'reasoning_quality', 'visual_understanding', 'response_clarity']
        
        for result in results:
            row = {
                'question_id': result['question_id'],
                'question': result['question'],
                'ground_truth': result['ground_truth'],
                'fot_prediction': result['fot_prediction'],
                'cot_prediction': result['cot_prediction'],
                'fot_correct': str(result['fot_prediction']).lower() == str(result['ground_truth']).lower(),
                'cot_correct': str(result['cot_prediction']).lower() == str(result['ground_truth']).lower(),
            }
            
            # 添加FOT分数
            for dim in dimensions:
                row[f'fot_{dim}'] = result['fot_scores'][dim]
                row[f'cot_{dim}'] = result['cot_scores'][dim]
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv('detailed_pope_evaluation_report.csv', index=False, encoding='utf-8')
        print("详细报告已保存至 detailed_pope_evaluation_report.csv")
        
        # 分析错误案例
        self.analyze_pope_errors(results)
    
    def generate_amber_report(self, results: List[Dict], output_file: str):
        """
        生成AMBER评估报告（无标准答案版本）
        """
        if not results:
            print("没有评估结果可生成报告")
            return
        
        # 检查数据结构
        if not results or 'evaluation' not in results[0]:
            print("评估结果数据结构不正确")
            return
        
        # 计算平均分数 - 修正数据结构访问
        fot_scores = {
            'reasoning_logic': [],
            'visual_understanding': [],
            'knowledge_application': [],
            'response_completeness': [],
            'total': []
        }
        
        cot_scores = {
            'reasoning_logic': [],
            'visual_understanding': [],
            'knowledge_application': [],
            'response_completeness': [],
            'total': []
        }
        
        for result in results:
            evaluation = result.get('evaluation', {})
            if 'fot_scores' in evaluation:
                for key in fot_scores.keys():
                    if key in evaluation['fot_scores']:
                        fot_scores[key].append(evaluation['fot_scores'][key])
            
            if 'cot_scores' in evaluation:
                for key in cot_scores.keys():
                    if key in evaluation['cot_scores']:
                        cot_scores[key].append(evaluation['cot_scores'][key])
        
        # 计算平均值
        fot_avg = {key: sum(scores)/len(scores) if scores else 0 for key, scores in fot_scores.items()}
        cot_avg = {key: sum(scores)/len(scores) if scores else 0 for key, scores in cot_scores.items()}
        
        # 生成报告
        report = {
            'evaluation_summary': {
                'total_evaluations': len(results),
                'fot_average_scores': fot_avg,
                'cot_average_scores': cot_avg,
                'fot_wins': sum(1 for r in results if r.get('evaluation', {}).get('fot_scores', {}).get('total', 0) > r.get('evaluation', {}).get('cot_scores', {}).get('total', 0)),
                'cot_wins': sum(1 for r in results if r.get('evaluation', {}).get('cot_scores', {}).get('total', 0) > r.get('evaluation', {}).get('fot_scores', {}).get('total', 0)),
                'ties': sum(1 for r in results if r.get('evaluation', {}).get('fot_scores', {}).get('total', 0) == r.get('evaluation', {}).get('cot_scores', {}).get('total', 0))
            },
            'detailed_results': results
        }
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"AMBER评估报告已保存到: {output_file}")
        print(f"FOT平均总分: {fot_avg['total']:.2f}")
        print(f"VoCoT平均总分: {cot_avg['total']:.2f}")
        print(f"FOT获胜: {report['evaluation_summary']['fot_wins']} 次")
        print(f"VoCoT获胜: {report['evaluation_summary']['cot_wins']} 次")
        print(f"平局: {report['evaluation_summary']['ties']} 次")
        
        # 修正维度列表 - 移除不存在的answer_accuracy
        dimensions = ['reasoning_logic', 'visual_understanding', 
                     'knowledge_application', 'response_completeness', 'total']
        
        total = len(results)
        
        print("\n=== AMBER评估报告 ===")
        print(f"总评估样本数: {total}")
        
        print("\n各维度平均分对比:")
        
        for dim in dimensions:
            if fot_scores[dim] and cot_scores[dim]:  # 确保有数据
                fot_dim_avg = sum(fot_scores[dim]) / len(fot_scores[dim])
                cot_dim_avg = sum(cot_scores[dim]) / len(cot_scores[dim])
                
                print(f"{dim}:")
                print(f"  FOT: {fot_dim_avg:.2f}")
                print(f"  VoCoT: {cot_dim_avg:.2f}")
                print(f"  差值: {fot_dim_avg - cot_dim_avg:+.2f}")
                print()
            else:
                print(f"{dim}: 数据不足")
        
        # 保存详细报告 - 修正数据访问
        df_data = []
        for result in results:
            evaluation = result.get('evaluation', {})
            row = {
                'item_id': result.get('item_id', ''),
                'question': result.get('question', ''),
                'ground_truth': result.get('ground_truth', ''),
                'fot_output': result.get('fot_output', ''),
                'cot_output': result.get('cot_output', ''),
            }
            
            # 添加分数 - 安全访问
            if 'fot_scores' in evaluation and 'cot_scores' in evaluation:
                for dim in dimensions:
                    row[f'fot_{dim}'] = evaluation['fot_scores'].get(dim, 0)
                    row[f'cot_{dim}'] = evaluation['cot_scores'].get(dim, 0)
            
            df_data.append(row)
        
        if df_data:
            df = pd.DataFrame(df_data)
            df.to_csv('detailed_amber_evaluation_report.csv', index=False, encoding='utf-8')
            print("详细报告已保存至 detailed_amber_evaluation_report.csv")
        
        # 分析优势案例
        self.analyze_amber_performance(results)

    def analyze_pope_errors(self, results: List[Dict]):
        """
        分析POPE错误案例
        """
        print("\n=== 错误案例分析 ===")
        
        fot_errors = []
        cot_errors = []
        
        for result in results:
            fot_correct = str(result['fot_prediction']).lower() == str(result['ground_truth']).lower()
            cot_correct = str(result['cot_prediction']).lower() == str(result['ground_truth']).lower()
            
            if not fot_correct:
                fot_errors.append(result)
            if not cot_correct:
                cot_errors.append(result)
        
        print(f"FOT错误案例数: {len(fot_errors)}")
        print(f"COT错误案例数: {len(cot_errors)}")
        
        # 分析FOT独有的错误
        fot_only_errors = [r for r in fot_errors if str(r['cot_prediction']).lower() == str(r['ground_truth']).lower()]
        cot_only_errors = [r for r in cot_errors if str(r['fot_prediction']).lower() == str(r['ground_truth']).lower()]
        
        print(f"FOT独有错误: {len(fot_only_errors)}")
        print(f"COT独有错误: {len(cot_only_errors)}")
        
        if fot_only_errors:
            print("\nFOT独有错误示例:")
            for i, error in enumerate(fot_only_errors[:3]):  # 显示前3个
                print(f"  {i+1}. 问题: {error['question']}")
                print(f"     标准答案: {error['ground_truth']}")
                print(f"     FOT预测: {error['fot_prediction']}")
                print(f"     COT预测: {error['cot_prediction']}")
                print()
    
    def analyze_amber_performance(self, results: List[Dict]):
        """
        分析AMBER性能表现
        """
        print("\n=== AMBER性能分析 ===")
        
        fot_errors = []
        cot_errors = []
        
        for result in results:
            # 对于AMBER数据集，使用更宽松的比较方式
            fot_prediction = str(result['fot_prediction']).strip().lower()
            cot_prediction = str(result['cot_prediction']).strip().lower()
            ground_truth = str(result['ground_truth']).strip().lower()
            
            # 简单的字符串匹配
            fot_correct = fot_prediction == ground_truth or ground_truth in fot_prediction
            cot_correct = cot_prediction == ground_truth or ground_truth in cot_prediction
            
            if not fot_correct:
                fot_errors.append(result)
            if not cot_correct:
                cot_errors.append(result)
        
        print(f"FOT错误案例数: {len(fot_errors)}")
        print(f"COT错误案例数: {len(cot_errors)}")
        
        # 分析FOT独有的错误
        fot_only_errors = []
        cot_only_errors = []
        
        for result in fot_errors:
            cot_prediction = str(result['cot_prediction']).strip().lower()
            ground_truth = str(result['ground_truth']).strip().lower()
            if cot_prediction == ground_truth or ground_truth in cot_prediction:
                fot_only_errors.append(result)
                
        for result in cot_errors:
            fot_prediction = str(result['fot_prediction']).strip().lower()
            ground_truth = str(result['ground_truth']).strip().lower()
            if fot_prediction == ground_truth or ground_truth in fot_prediction:
                cot_only_errors.append(result)
        
        print(f"FOT独有错误: {len(fot_only_errors)}")
        print(f"COT独有错误: {len(cot_only_errors)}")
        
        # 分析各维度的优势
        dimensions = ['answer_accuracy', 'reasoning_logic', 'visual_understanding', 
                     'knowledge_application', 'response_completeness']
        
        for dim in dimensions:
            fot_wins = sum(1 for r in results if r['fot_scores'][dim] > r['cot_scores'][dim])
            cot_wins = sum(1 for r in results if r['cot_scores'][dim] > r['fot_scores'][dim])
            ties = len(results) - fot_wins - cot_wins
            
            print(f"{dim}:")
            print(f"  FOT胜出: {fot_wins} ({fot_wins/len(results)*100:.1f}%)")
            print(f"  COT胜出: {cot_wins} ({cot_wins/len(results)*100:.1f}%)")
            print(f"  平局: {ties} ({ties/len(results)*100:.1f}%)")
            print()
        
        # 分析FOT思维过程的影响
        fot_with_thoughts = [r for r in results if r.get('fot_scores', {}).get('reasoning_logic', 0) > 7]
        print(f"FOT推理质量高分(>7)案例: {len(fot_with_thoughts)} ({len(fot_with_thoughts)/len(results)*100:.1f}%)")

    # 保留原有方法以兼容其他任务
    def run_evaluation(self, baseline_file: str, fot_file: str, 
                      output_file: str = "evaluation_results.json"):
        """
        运行通用的批量评估
        """
        print("开始评估...")
        
        # 解析输出文件
        baseline_results = self.processor.parse_baseline_output(baseline_file)
        fot_results = self.processor.parse_fot_output(fot_file)
        
        # 匹配相同问题的结果
        matched_pairs = self.match_results(baseline_results, fot_results)
        
        if not matched_pairs:
            print("没有找到匹配的问题对")
            return []
        
        evaluation_results = []
        
        # 逐个评估
        for i, (baseline_item, fot_item) in enumerate(matched_pairs):
            print(f"评估进度: {i+1}/{len(matched_pairs)}")
            
            # 调用评估API
            eval_result = self.evaluator.evaluate_outputs(
                question=baseline_item['question'],
                ground_truth=baseline_item['ground_truth'],
                baseline_output=baseline_item['prediction'],
                fot_output=fot_item['prediction'],
                fot_thoughts=fot_item.get('thoughts', [])
            )
            
            if eval_result:
                # 整合结果
                result = {
                    'question_id': baseline_item['question_id'],
                    'question': baseline_item['question'],
                    'ground_truth': baseline_item['ground_truth'],
                    'baseline_prediction': baseline_item['prediction'],
                    'fot_prediction': fot_item['prediction'],
                    'baseline_scores': eval_result['baseline_scores'],
                    'fot_scores': eval_result['fot_scores'],
                    'detailed_analysis': eval_result['detailed_analysis']
                }
                evaluation_results.append(result)
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
            
        # 生成统计报告
        self.generate_report(evaluation_results)
        
        return evaluation_results
        
    def match_results(self, baseline: List[Dict], fot: List[Dict]) -> List[tuple]:
        """
        匹配相同问题的结果
        """
        baseline_dict = {item['question_id']: item for item in baseline}
        matched_pairs = []
        
        for fot_item in fot:
            qid = fot_item['question_id']
            if qid in baseline_dict:
                matched_pairs.append((baseline_dict[qid], fot_item))
                
        print(f"成功匹配 {len(matched_pairs)} 个问题对")
        return matched_pairs
    
    def generate_report(self, results: List[Dict]):
        """
        生成评估报告
        """
        if not results:
            print("没有评估结果")
            return
            
        # 计算平均分
        baseline_scores = []
        fot_scores = []
        
        for result in results:
            baseline_scores.append(result['baseline_scores'])
            fot_scores.append(result['fot_scores'])
        
        # 计算各维度平均分
        dimensions = ['object_similarity', 'logical_completeness', 
                     'image_description', 'thinking_comprehensiveness', 'total']
        
        print("\n=== 评估报告 ===")
        print(f"总评估样本数: {len(results)}")
        
        for dim in dimensions:
            baseline_avg = sum(score[dim] for score in baseline_scores) / len(baseline_scores)
            fot_avg = sum(score[dim] for score in fot_scores) / len(fot_scores)
            
            print(f"{dim}:")
            print(f"  Baseline: {baseline_avg:.2f}")
            print(f"  FOT: {fot_avg:.2f}")
            print(f"  差值: {fot_avg - baseline_avg:+.2f}")
            print()
        
        # 保存详细报告
        df_data = []
        for result in results:
            row = {
                'question_id': result['question_id'],
                'question': result['question'],
                'ground_truth': result['ground_truth'],
                'baseline_prediction': result['baseline_prediction'],
                'fot_prediction': result['fot_prediction'],
            }
            
            # 添加分数
            for dim in dimensions:
                row[f'baseline_{dim}'] = result['baseline_scores'][dim]
                row[f'fot_{dim}'] = result['fot_scores'][dim]
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv('detailed_evaluation_report.csv', index=False, encoding='utf-8')