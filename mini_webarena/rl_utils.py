import os
import re
import csv
import json
from collections import defaultdict

def format_score(s: str, is_success: bool = True) -> float:
    """
    给定字符串 s，基于以下规则进行打分:
        1) 有 <think> 标签(同时也要有 </think>) => 0.3
           如果没有, 直接返回 0 分。
        2) <think> 前面无内容(仅空白) => 0.1
        3) `</think>` 后面能匹配到一个动作 => 0.2
        4) 正好只有这个动作(没有多余字符) => 0.2
        5) 动作可被执行 (is_success=True) => 0.2
    """

    score = 0.0

    # 1) 有 <think> 且有 </think>
    if "<think>" not in s or "</think>" not in s:
        return 0.0
    score += 0.3

    # 2) <think> 前面无内容(忽略空白)
    idx_think = s.index("<think>")
    prefix = s[:idx_think].strip()
    if prefix == "":
        score += 0.1

    # 开始准备匹配动作
    # 假定用三重反引号 ``` 包裹动作，形如:
    #   ```动作内容```
    # 如果你的环境使用其他分隔符，需要自行替换这里的正则
    # 先抽取 <think> 后面的内容 (tail_content)
    # 以便检查动作
    tail_part = s.split("</think>", maxsplit=1)
    if len(tail_part) < 2:
        # 没有后半部分，也就没有动作
        # 依然可能有第 5 条加分，不过要看需求，一般不会给
        # 这里我们先不加分
        # 最后再加上 is_success 的加分(但如果动作都没匹配到, is_success是否还有意义? 视需求而定)
        if is_success:
            score += 0.2
        return round(score, 3)

    tail_content = tail_part[1].strip()

    # 3) `</think>` 后面是否能找到一个动作(先尝试正则提取第一个动作)
    #    使用非贪婪匹配，在 tail_content 中找第一个三重反引号包裹的内容
    pattern = r"```((.|\n)*?)```"
    match = re.search(pattern, tail_content)
    if match:
        # 找到了动作 => +0.2
        score += 0.2
        # 提取动作文本(去除首尾空白)
        action_text = match.group(1).strip()

        # 4) 检查是否「正好只有这一个动作」
        #    也就是说 tail_content 必须完全等于 ```{action_text}```(忽略首尾空白)
        #    这里可以构造一个期望的文本块，然后与 tail_content 比较
        expected_block = f"```{action_text}```"
        if tail_content == expected_block:
            score += 0.2
    else:
        # 没能匹配到动作，不加这 0.2
        # 也没有第4条的可能了
        pass

    # 5) 动作可被执行
    if is_success:
        score += 0.2

    return round(score, 3)


def process_data(path=None):
    """
    读取 path 指定的 JSON 文件，返回一个 list(dict)，其中每个 dict 包含:
    {
        "task_id": str,
        "sample_id": str,
        "step_num": int,
        "input": str,
        "output": str,
        "answer_score": float,
        "format_score": float
    }

    注意:
    1. task_id, sample_id 从文件路径推断；
    2. step_num = 这是第几个 action (按 trajectory 的顺序计数)；
    3. input = item["prompt"]；
    4. output = item["raw_prediction"]；
    5. answer_score = 最外层 data["score"] (需要 float 转换)；
    6. format_score = 依据 is_success 判断, 其中 is_success = (action_type != "ACTION_TYPES.NONE")。
    """
    if path is None:
        return []

    # 根据文件路径提取 task_id, sample_id
    filename = os.path.basename(path)           # e.g. "result_25.json"
    dir_name = os.path.dirname(path)            # e.g. ".../task_1"
    task_id = os.path.basename(dir_name)        # e.g. "task_1"
    sample_id = filename.replace("result_", "").replace(".json", "")

    # 读取 JSON 文件
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 外层 answer_score
    answer_score = float(data.get("score", 0.0))

    trajectory = data.get("trajectory", [])
    results = []

    step_num = 0
    # step_num 计数
    for idx, item in enumerate(trajectory):
        # 如果这条记录里没有 action_type，就跳过
        if "action_type" not in item:
            continue

        step_num += 1  # 第几步 action = 在 trajectory 中的索引
        input_text = item.get("prompt", "")
        output_text = item.get("raw_prediction", "")

        # 如果 action_type == "ACTION_TYPES.NONE" 则视为失败
        action_type = item["action_type"]
        is_success = (action_type != "ACTION_TYPES.NONE")
        f_score = format_score(output_text, is_success=is_success)
        if output_text == "": # Skip the system terminated step
            continue

        # 整合记录
        results.append({
            "task_id": task_id,
            "sample_id": sample_id,
            "step_num": step_num,
            "input": input_text,
            "output": output_text,
            "answer_score": answer_score,
            "format_score": f_score,
        })
    # print("Processed", len(results), "records")
    # print("Path:", path)
    # # print("Answer score:", [r["answer_score"] for r in results])
    # # print("Format score:", [r["format_score"] for r in results])
    # for r in results:
    #     print(f"Step {r['step_num']}: {r['input']}")
    #     print(f"Output: {r['output']}")
    #     print(f"Answer score: {r['answer_score']}, Format score: {r['format_score']}")
    #     print()
    # exit(0)
    return results

def visualize_result():
    """
    1. 遍历 ray 目录下各个 task_xxx 子目录；
    2. 收集并写入 rl_data/reward_data.csv；
    3. 统计并写入 rl_data/dp_data.csv；
    """
    base_dir = "ray"
    output_dir = "rl_data"
    os.makedirs(output_dir, exist_ok=True)

    reward_csv_path = os.path.join(output_dir, "reward_data.csv")
    dp_csv_path = os.path.join(output_dir, "dp_data.csv")

    reward_headers = ["task_id", "sample_id", "step_num", "input", "output", "answer_score", "format_score"]
    dp_headers = ["task_id", "sample_size", "acc"]

    dp_data = defaultdict(lambda: {"samples": {}})

    with open(reward_csv_path, "w", newline="", encoding="utf-8") as reward_file:
        writer = csv.DictWriter(reward_file, fieldnames=reward_headers)
        writer.writeheader()

        for task_dir in os.listdir(base_dir):
            task_path = os.path.join(base_dir, task_dir)
            if not os.path.isdir(task_path):
                continue  # 如果不是目录，就跳过
            task_id = task_dir
            for filename in os.listdir(task_path):
                if not (filename.startswith("result_") and filename.endswith(".json")):
                    continue
                sample_id = filename.replace("result_", "").replace(".json", "")
                file_path = os.path.join(task_path, filename)
                records = process_data(file_path)
                for record in records:
                    row = {
                        "task_id": task_id,
                        "sample_id": sample_id,
                        "step_num": record["step_num"],
                        "input": record["input"],
                        "output": record["output"],
                        "answer_score": record["answer_score"],
                        "format_score": record["format_score"],
                    }
                    writer.writerow(row)
                    dp_data[task_id]["samples"][sample_id] = record["answer_score"]
    with open(dp_csv_path, "w", newline="", encoding="utf-8") as dp_file:
        writer = csv.DictWriter(dp_file, fieldnames=dp_headers)
        writer.writeheader()
        for task_id, info_dict in dp_data.items():
            samples = info_dict["samples"]
            sample_size = len(samples)
            if sample_size == 0:
                continue
            sum_score = sum(samples[s_id] for s_id in samples)
            acc = sum_score / sample_size
            writer.writerow({
                "task_id": task_id,
                "sample_size": sample_size,
                "acc": acc
            })

def test_format_score():
    # 1) 完整示例: 预期得分 1.0
    #  - 有 <think> => 0.3
    #  - <think> 前面无内容 => 0.1
    #  - 能解析出动作 => 0.2
    #  - 正好只有这一个动作 => 0.2
    #  - is_success=True => 0.2
    #  合计 1.0
    s1 = """<think> The objective is to search who replaced Bradie Tennell in the 2021 Skate America. </think>
```type [6] [who replaced Bradie Tennell in 2021 Skate America] [1]```"""
    score1 = format_score(s1, is_success=True)
    print("Test1 score =", score1, "(expected 1.0)")

    # 2) 没有 <think>: 预期得分 0
    s2 = """Hello
```type [6] something [1]```"""
    score2 = format_score(s2, is_success=True)
    print("Test2 score =", score2, "(expected 0.0)")

    # 3) <think> 前有内容: 预期少 0.1
    #  - 有 <think> => +0.3
    #  - <think> 前面有文本 => 不加 0.1
    #  - 动作正常 => +0.2
    #  - 若正好只有这一个动作 => +0.2
    #  - is_success=True => +0.2
    #  最终应该 = 0.3 + 0.2 + 0.2 + 0.2 = 0.9
    s3 = """Some text before <think> I'm thinking. </think>
```type [6] some action [1]```"""
    score3 = format_score(s3, is_success=True)
    print("Test3 score =", score3, "(expected 0.9)")

    # 4) 有 <think> 但提取不到动作: 预期 0.3 + 0.1 + 0 (没动作) + 0 (没动作) + 0.2 = 0.6
    s4 = """<think> no action after think </think>
No triple backticks here..."""
    score4 = format_score(s4, is_success=True)
    print("Test4 score =", score4, "(expected 0.6)")

    # 5) 有两个动作，理论上只能提取到第一个，但多余内容会导致失去第4条额外 0.2
    #  - 有 <think> => 0.3
    #  - <think> 前面无内容 => 0.1
    #  - 可以解析第一个动作 => +0.2
    #  - 但不是“正好”只有这一条动作 => 不加 0.2
    #  - is_success=True => 0.2
    #  => 总共 1.0 - 0.2 = 0.8
    s5 = """<think> I'm thinking. </think>
```type [6] first action [1]```
```type [7] second action [1]```"""
    score5 = format_score(s5, is_success=True)
    print("Test5 score =", score5, "(expected 0.8)")

if __name__ == "__main__":
    visualize_result()