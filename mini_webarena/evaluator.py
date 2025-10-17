import nltk
import json
from nltk.tokenize import word_tokenize
from difflib import SequenceMatcher


# 如果还没下载 'punkt':
# nltk.download('punkt')

def clean_text(text: str) -> str:
    text = text.strip().lower()
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        text = text[1:-1]
    return text


def char_lcs_ratio(ref: str, pred: str) -> float:
    """
    计算字符级 LCS 长度除以较长串长度，用 difflib.SequenceMatcher 实现。
    LCS 越长，说明顺序和内容越相似。
    """
    matcher = SequenceMatcher(None, ref, pred)
    lcs_len = sum(block.size for block in matcher.get_matching_blocks())
    max_len = max(len(ref), len(pred)) or 1
    return lcs_len / max_len


def token_f1(ref: str, pred: str) -> float:
    """
    计算词级别的 F1 分数，不考虑顺序。
    - P = 交集 / pred_tokens
    - R = 交集 / ref_tokens
    - F1 = 2 * (P * R) / (P + R)
    """
    ref_tokens = set(word_tokenize(ref))
    pred_tokens = set(word_tokenize(pred))

    if not ref_tokens or not pred_tokens:
        return 0.0

    intersection = ref_tokens & pred_tokens
    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(ref_tokens)

    if precision == 0 and recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def edit_distance_ratio(ref: str, pred: str) -> float:
    """
    计算字符级编辑距离 / max_len，作为惩罚项（越小越好）。
    """
    # 简单 Levenshtein 可用 nltk 或 textdistance，这里演示 difflib。
    matcher = SequenceMatcher(None, ref, pred)
    # difflib 没有直接给出 edit distance，可以用 ratio 辅助或写一个简单 Levenshtein
    # 为了示例，我们先手写一个最简单的版本
    # ---- 手写 Levenshtein 距离实现 (更适用于短文本) ----

    # 如果字符串很长，可考虑其他更快算法
    dp = [[0] * (len(pred) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(pred) + 1):
        dp[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(pred) + 1):
            cost = 0 if ref[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # 删除
                dp[i][j - 1] + 1,  # 插入
                dp[i - 1][j - 1] + cost  # 替换
            )

    edit_dist = dp[len(ref)][len(pred)]
    max_len = max(len(ref), len(pred)) or 1
    return edit_dist / max_len


def fuzzy_match(ref: str, pred: str,
                alpha: float = 0.7,
                beta: float = 0.3,
                gamma: float = 0.1) -> float:
    """
    计算最终分数:
    1. 字符级 LCS (考虑顺序)       => char_lcs_ratio
    2. 词级别 F1 (不考虑顺序)      => token_f1
    3. 编辑距离惩罚 (越高惩罚越大) => edit_distance_ratio

    返回公式:
    score = alpha * char_lcs + beta * token_f1 - gamma * edit_distance

    默认 alpha=0.7, beta=0.3, gamma=0.3:
    - 完全匹配 => ~1
    - 顺序调换 => ~0.6 ~ 0.7
    - 仅部分相同 => ~0.2
    """
    # print("Now is in fuzzy_match")
    # print(f"ref: {ref}, pred: {pred}")

    ref = clean_text(ref)
    pred = clean_text(pred)

    char_lcs = char_lcs_ratio(ref, pred)  # [0, 1]
    tok_f1 = token_f1(ref, pred)  # [0, 1]
    dist_penalty = edit_distance_ratio(ref, pred)  # [0, 1+]

    score = alpha * char_lcs + beta * tok_f1 - gamma * dist_penalty
    # print("score: ", score)
    return max(0.0, min(score, 1.0))  # 可以截断在 [0, 1] 内，视需求而定

# Not Used
def get_last_action(trajectory):
    if not trajectory:
        raise ValueError("Trajectory is empty, cannot get last action.")
    return trajectory[-1]


# Not Used
def compute_score_with_fuzzy_match(trajectory, config_file) -> float:
    """
    用 fuzzy_match 计算分数的最小示例。
    """
    # 1. 读取配置，获取参考答案
    with open(config_file, "r") as f:
        config = json.load(f)

    # 假设 reference_answers 是一个单独字符串，比如 "Starr Andrews"
    # 如果它是数组，你可以自行处理
    reference_answer = config["eval"]["reference_answers"]

    # 2. 获取最后一个 Action 中的 pred
    last_action = get_last_action(trajectory)
    pred = last_action["answer"]  # 根据你之前的逻辑

    # 3. 计算最终分数
    scores = []
    for ref in reference_answer:
        score = fuzzy_match(ref, pred)
        scores.append(score)
    return max(scores)

# ========== 测试示例 ==========
def metric_exact_match(refs, pred):
    norm_pred = pred.strip().lower()
    norm_refs = [r.strip().lower() for r in refs]
    return 1 if (norm_pred in norm_refs) else 0

def metric_heuristic(refs, pred):
    norm_pred = pred.strip().lower()
    norm_refs = [r.strip().lower() for r in refs]
    return max(fuzzy_match(ref, norm_pred) for ref in norm_refs)

if __name__ == '__main__':
    # nltk.download('punkt', quiet=True)
    # nltk.download('punkt_tab', quiet=True)
    #
    # ref = "Starr Andrews"
    # pred1 = "Starr Andrews"  # 完全相同
    # pred2 = "Andrews Starr"  # 顺序调换
    # pred3 = "Andrews John"  # 仅部分匹配
    #
    # s1 = fuzzy_match(ref, pred1)  # ~1
    # s2 = fuzzy_match(ref, pred2)  # ~0.6
    # s3 = fuzzy_match(ref, pred3)  # ~0.2
    #
    # print(f"Pred1 => {s1:.4f}")
    # print(f"Pred2 => {s2:.4f}")
    # print(f"Pred3 => {s3:.4f}")
    refs = ["of Luca Pacioli", "Summa de arithmetica"]
    pred = "of Luca Pacioli"
    print(f"Example 1, refs = {refs}, pred = {pred}")
    print("Exact match: ", metric_exact_match(refs, pred))
    print("Heuristic: ", metric_heuristic(refs, pred))
    print("")
    refs = ["of Luca Pacioli", "Summa de arithmetica"]
    pred = "Luca Pacioli"
    print(f"Example 2, refs = {refs}, pred = {pred}")
    print("Exact match: ", metric_exact_match(refs, pred))
    print("Heuristic: ", metric_heuristic(refs, pred))
    print("")
    refs = ["of Luca Pacioli", "Summa de arithmetica"]
    pred = "Pacioli Luca"
    print(f"Example 3, refs = {refs}, pred = {pred}")
    print("Exact match: ", metric_exact_match(refs, pred))
    print("Heuristic: ", metric_heuristic(refs, pred))