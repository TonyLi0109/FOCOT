import os

from batch_evaluation import BatchEvaluator

# 配置API密钥
API_KEY = os.environ["OPENAI_API_KEY"]

# 创建评估器
evaluator = BatchEvaluator(API_KEY)

# 运行评估
baseline_output_file = "output/gqa/baseline_model/predictions.json"
fot_output_file = "output/gqa/fot_model/detailed_output.txt"

evaluator.run_evaluation(
    baseline_file=baseline_output_file,
    fot_file=fot_output_file,
    output_file="gqa_evaluation_results.json"
)
