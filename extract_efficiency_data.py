#!/usr/bin/env python3
"""
FOCOT Efficiency Index 数据提取脚本

用途: 从现有的评估结果中快速提取和组织性能数据用于论文修订
输出: 包含所有效率指标的结构化报告

使用: python extract_efficiency_data.py
"""

import json
import csv
import statistics
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    # 推理时间指标
    inference_time_mean: float = 0.0      # 平均推理延迟 (秒)
    inference_time_std: float = 0.0       # 推理延迟标准差
    throughput_mean: float = 0.0          # 平均吞吐量 (tokens/s)
    throughput_std: float = 0.0           # 吞吐量标准差
    
    # GPU内存指标
    gpu_memory_peak: float = 0.0          # GPU峰值内存 (GB)
    gpu_memory_avg: float = 0.0           # 平均GPU内存 (GB)
    memory_efficiency: float = 0.0        # 内存效率 (tokens/GB)
    
    # 准确率指标
    accuracy: float = 0.0                 # 准确率 (%)
    accuracy_improvement: float = 0.0     # 相对改进 (%)
    
    # 评分指标
    reasoning_logic_score: float = 0.0    # 推理逻辑性评分 (1-10)
    visual_understanding_score: float = 0.0  # 视觉理解评分
    knowledge_application_score: float = 0.0  # 知识运用评分
    completeness_score: float = 0.0       # 完整性评分
    overall_score: float = 0.0            # 总分
    
    # 胜出率
    win_rate: float = 0.0                 # 胜出率 (%)
    
    # 效率指数
    efficiency_index: float = 0.0         # 综合效率指数 (0-100)


class EfficiencyDataExtractor:
    """从评估结果中提取效率数据"""
    
    def __init__(self, work_dir: str = "."):
        self.work_dir = Path(work_dir)
        
    def load_evaluation_results(self, filepath: str) -> List[Dict]:
        """加载评估结果JSON文件"""
        with open(self.work_dir / filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_fot_metrics(self, results: List[Dict]) -> Dict:
        """从FOT评估结果中提取指标"""
        metrics = {}
        
        # 提取准确率
        correct = sum(1 for r in results if r.get('fot_prediction', '').lower() 
                     == r.get('ground_truth', '').lower())
        metrics['fot_accuracy'] = (correct / len(results) * 100) if results else 0
        
        # 提取评分（如果存在）
        if results and 'evaluation' in results[0]:
            fot_scores = [r['evaluation']['fot_scores'] 
                         for r in results if 'evaluation' in r]
            if fot_scores:
                metrics['reasoning_logic'] = statistics.mean(
                    [s['reasoning_logic'] for s in fot_scores])
                metrics['visual_understanding'] = statistics.mean(
                    [s['visual_understanding'] for s in fot_scores])
                metrics['knowledge_application'] = statistics.mean(
                    [s['knowledge_application'] for s in fot_scores])
                metrics['completeness'] = statistics.mean(
                    [s['response_completeness'] for s in fot_scores])
                metrics['overall_score'] = statistics.mean(
                    [s.get('total', 0) for s in fot_scores])
        
        return metrics
    
    def extract_cot_metrics(self, results: List[Dict]) -> Dict:
        """从COT评估结果中提取指标"""
        metrics = {}
        
        # 提取准确率
        correct = sum(1 for r in results if r.get('cot_prediction', '').lower() 
                     == r.get('ground_truth', '').lower())
        metrics['cot_accuracy'] = (correct / len(results) * 100) if results else 0
        
        # 提取评分（如果存在）
        if results and 'evaluation' in results[0]:
            cot_scores = [r['evaluation']['cot_scores'] 
                         for r in results if 'evaluation' in r]
            if cot_scores:
                metrics['reasoning_logic'] = statistics.mean(
                    [s['reasoning_logic'] for s in cot_scores])
                metrics['visual_understanding'] = statistics.mean(
                    [s['visual_understanding'] for s in cot_scores])
                metrics['knowledge_application'] = statistics.mean(
                    [s['knowledge_application'] for s in cot_scores])
                metrics['completeness'] = statistics.mean(
                    [s['response_completeness'] for s in cot_scores])
                metrics['overall_score'] = statistics.mean(
                    [s.get('total', 0) for s in cot_scores])
        
        return metrics
    
    def calculate_efficiency_index(self, fot_metrics: Dict, cot_metrics: Dict) -> float:
        """计算综合效率指数 (0-100)"""
        
        # 准确率改进（规范化到0-1）
        accuracy_improvement = (fot_metrics.get('fot_accuracy', 0) - 
                               cot_metrics.get('cot_accuracy', 0))
        accuracy_norm = min(accuracy_improvement / 20, 1.0)  # 20%改进为上限
        
        # 质量改进（规范化到0-1）
        fot_quality = (fot_metrics.get('reasoning_logic', 0) +
                      fot_metrics.get('visual_understanding', 0) +
                      fot_metrics.get('knowledge_application', 0) +
                      fot_metrics.get('completeness', 0)) / 4
        
        cot_quality = (cot_metrics.get('reasoning_logic', 0) +
                      cot_metrics.get('visual_understanding', 0) +
                      cot_metrics.get('knowledge_application', 0) +
                      cot_metrics.get('completeness', 0)) / 4
        
        quality_improvement = fot_quality - cot_quality
        quality_norm = min((quality_improvement / 4), 1.0)  # 4分改进为上限
        
        # 计算综合指数
        efficiency_index = (
            accuracy_norm * 0.40 +           # 准确率: 40%
            quality_norm * 0.40 +            # 质量: 40%
            0.15 -                           # 效率: 15% (需要实际数据)
            0.05                             # 资源成本: -5% (需要实际数据)
        )
        
        # 标准化到0-100
        return max(0, min(100, efficiency_index * 100))
    
    def generate_comparison_table(self, fot_metrics: Dict, cot_metrics: Dict) -> str:
        """生成性能对比表格"""
        
        table = """
| 指标 | FOT | COT | 改进 |
|------|-----|-----|------|
"""
        
        # 准确率行
        fot_acc = fot_metrics.get('fot_accuracy', 0)
        cot_acc = cot_metrics.get('cot_accuracy', 0)
        improvement = ((fot_acc - cot_acc) / cot_acc * 100) if cot_acc > 0 else 0
        table += f"| 准确率 (%) | {fot_acc:.2f}% | {cot_acc:.2f}% | +{improvement:.2f}% |\n"
        
        # 推理逻辑性
        fot_logic = fot_metrics.get('reasoning_logic', 0)
        cot_logic = cot_metrics.get('reasoning_logic', 0)
        table += f"| 推理逻辑性 (1-10) | {fot_logic:.2f} | {cot_logic:.2f} | +{fot_logic-cot_logic:.2f} |\n"
        
        # 视觉理解
        fot_visual = fot_metrics.get('visual_understanding', 0)
        cot_visual = cot_metrics.get('visual_understanding', 0)
        table += f"| 视觉理解 (1-10) | {fot_visual:.2f} | {cot_visual:.2f} | +{fot_visual-cot_visual:.2f} |\n"
        
        # 知识运用
        fot_knowledge = fot_metrics.get('knowledge_application', 0)
        cot_knowledge = cot_metrics.get('knowledge_application', 0)
        table += f"| 知识运用 (1-10) | {fot_knowledge:.2f} | {cot_knowledge:.2f} | +{fot_knowledge-cot_knowledge:.2f} |\n"
        
        # 完整性
        fot_complete = fot_metrics.get('completeness', 0)
        cot_complete = cot_metrics.get('completeness', 0)
        table += f"| 完整性 (1-10) | {fot_complete:.2f} | {cot_complete:.2f} | +{fot_complete-cot_complete:.2f} |\n"
        
        # 总分
        fot_total = fot_metrics.get('overall_score', 0)
        cot_total = cot_metrics.get('overall_score', 0)
        table += f"| 综合评分 (1-10) | {fot_total:.2f} | {cot_total:.2f} | +{fot_total-cot_total:.2f} |\n"
        
        return table
    
    def generate_report(self, output_file: str = "efficiency_report.txt"):
        """生成完整的效率分析报告"""
        
        try:
            # 加载数据
            print("正在加载评估数据...")
            results = self.load_evaluation_results("amber_evaluation_results.json")
            
            # 提取指标
            print("正在提取指标...")
            fot_metrics = self.extract_fot_metrics(results)
            cot_metrics = self.extract_cot_metrics(results)
            
            # 计算效率指数
            print("正在计算效率指数...")
            efficiency_index = self.calculate_efficiency_index(fot_metrics, cot_metrics)
            
            # 生成报告
            report = []
            report.append("=" * 60)
            report.append("FOCOT Efficiency Index 分析报告")
            report.append("=" * 60)
            report.append("")
            
            # 摘要部分
            report.append("【摘要】")
            report.append(f"样本数: {len(results)}")
            report.append(f"FOT准确率: {fot_metrics.get('fot_accuracy', 0):.2f}%")
            report.append(f"COT准确率: {cot_metrics.get('cot_accuracy', 0):.2f}%")
            report.append(f"准确率改进: +{(fot_metrics.get('fot_accuracy', 0) - cot_metrics.get('cot_accuracy', 0)):.2f}%")
            report.append(f"综合效率指数: {efficiency_index:.2f}/100")
            report.append("")
            
            # 性能对比表
            report.append("【性能对比表】")
            report.append(self.generate_comparison_table(fot_metrics, cot_metrics))
            report.append("")
            
            # 详细指标
            report.append("【详细指标分析】")
            report.append("")
            report.append("FOT指标:")
            for key, value in fot_metrics.items():
                report.append(f"  {key}: {value:.4f}")
            report.append("")
            report.append("COT指标:")
            for key, value in cot_metrics.items():
                report.append(f"  {key}: {value:.4f}")
            report.append("")
            
            # 效率指数计算
            report.append("【效率指数计算】")
            fot_quality = sum([fot_metrics.get(k, 0) for k in 
                             ['reasoning_logic', 'visual_understanding', 
                              'knowledge_application', 'completeness']]) / 4
            cot_quality = sum([cot_metrics.get(k, 0) for k in 
                             ['reasoning_logic', 'visual_understanding', 
                              'knowledge_application', 'completeness']]) / 4
            accuracy_delta = fot_metrics.get('fot_accuracy', 0) - cot_metrics.get('cot_accuracy', 0)
            quality_delta = fot_quality - cot_quality
            
            report.append(f"准确率增长: {accuracy_delta:+.2f}% (权重: 40%)")
            report.append(f"质量增长: {quality_delta:+.2f}分 (权重: 40%)")
            report.append(f"效率增长: 待测量 (权重: 15%)")
            report.append(f"资源成本: 待测量 (权重: -5%)")
            report.append(f"最终 Efficiency Index: {efficiency_index:.2f}/100")
            report.append("")
            
            # 建议
            report.append("【论文修订建议】")
            report.append("1. 在第4.3节添加计算效率分析小节")
            report.append("2. 新增表格显示性能对比数据")
            report.append("3. 补充GPU时间、内存使用等具体数据")
            report.append("4. 添加消融实验验证效率指数的各个组成部分")
            report.append("")
            
            # 保存报告
            report_text = "\n".join(report)
            with open(self.work_dir / output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            print(f"✅ 报告已保存到: {output_file}")
            print(report_text)
            
            return report_text
            
        except FileNotFoundError as e:
            print(f"❌ 文件未找到: {e}")
            print("请确保以下文件存在:")
            print("  - amber_evaluation_results.json")
            return None
        except Exception as e:
            print(f"❌ 错误: {e}")
            return None


def main():
    """主函数"""
    import sys
    
    # 使用当前目录
    work_dir = Path.cwd()
    extractor = EfficiencyDataExtractor(str(work_dir))
    
    print(f"工作目录: {work_dir}")
    print("="*60)
    
    # 生成报告
    extractor.generate_report("efficiency_index_report.txt")
    
    print("="*60)
    print("✅ 数据提取完成！")
    print("\n后续步骤:")
    print("1. 查看生成的 efficiency_index_report.txt")
    print("2. 根据实际的GPU时间和内存数据完善报告")
    print("3. 将数据插入论文的第4.3计算效率分析节")


if __name__ == "__main__":
    main()
