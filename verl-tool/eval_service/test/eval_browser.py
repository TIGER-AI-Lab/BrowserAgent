import pandas as pd
from tqdm import trange
import argparse
import os
from mini_webarena.evaluator import metric_exact_match, metric_heuristic

def extract_pred_from_output(output):
    """
    从output中提取最后一个 stop [xxx] 里的 xxx 作为 pred。
    """
    import re
    if not isinstance(output, str):
        return ""
    # 匹配所有 stop [xxx]，取最后一个
    pattern = r'```stop \[(.*?)\]'
    matches = re.findall(pattern, output)
    if matches:
        return matches[-1].strip()
    else:
        return ""

def eval_inference(result_path, save_path=None):
    """
    输入csv路径，评测output和golden_answers的em/heur分数，输出带分数的csv
    """
    result_df = pd.read_csv(result_path)
    # 兼容golden_answers为字符串list的情况
    def parse_golden(x):
        import ast
        if pd.isna(x):
            return []
        if isinstance(x, list):
            return x
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple)):
                return list(v)
            else:
                return [str(v)]
        except Exception:
            return [str(x)]
    gts = result_df['golden_answers'].apply(parse_golden)
    # preds = result_df['output'].fillna("").astype(str)
    preds = result_df['output'].fillna("").astype(str).apply(extract_pred_from_output)
    em_scores = []
    heur_scores = []
    for i in trange(len(result_df), desc="Scoring"):
        em_scores.append(metric_exact_match(gts.iloc[i], preds.iloc[i]))
        heur_scores.append(metric_heuristic(gts.iloc[i], preds.iloc[i]))
    result_df['em'] = em_scores
    result_df['heur'] = heur_scores
    em_avg = sum(em_scores) / len(em_scores) if em_scores else 0
    heur_avg = sum(heur_scores) / len(heur_scores) if heur_scores else 0
    print(f"Total samples: {len(result_df)}")
    print(f"Exact Match Score (em): {em_avg:.6f}")
    print(f"Heuristic Score (heur): {heur_avg:.6f}")
    if save_path is None:
        return
        save_path = result_path.replace('.csv', '_eval.csv')
    result_df.to_csv(save_path, index=False)
    print(f"评测结果已保存到 {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_csv', type=str, default="/home/yutao/verl-tool/eval_service/result/Qwen2.5-3B-Instruct_result.csv", help='inference_browserAPI生成的csv路径')
    parser.add_argument('--save_path', type=str, default="/home/yutao/verl-tool/eval_service/result/Qwen2.5-3B-Instruct_result_eval.csv", help='评测结果保存路径')
    args = parser.parse_args()
    eval_inference(args.result_csv, args.save_path)
    # print(extract_pred_from_output("```stop [22.0.4]```"))