import re
import json


def parse_log_entry(log_text):
    """解析单个日志条目并转换为字典"""
    data = {}

    # 定义正则表达式来提取不同内容
    quant_model_path = re.search(r"Loading quantized model from (.+)", log_text)
    amp_enabled = re.search(r"AMP enabled: (True|False)", log_text)
    wikitext2_score = re.search(r"wikitext2 perplexity: ([\d.]+)", log_text)
    c4_score = re.search(r"c4 perplexity: ([\d.]+)", log_text)
    average_acc = re.search(r"Average Acc: ([\d.]+)%", log_text)
    
    # 任务表格数据
    tasks = []
    task_pattern = re.compile(
        r"\|([a-z_]+)\s*\|\s*(\d+)\|none\s*\|\s*(\d+)\|(\w+)\s*\|(\s*↑?\s*)\|?([\d.]+)\|\s*±\s*\|([\d.]+)\|"
    )

    # 提取模型路径和数据
    data["quant_model_path"] = quant_model_path.group(1) if quant_model_path else None
    data["amp_enabled"] = amp_enabled.group(1) == "True" if amp_enabled else None
    data["wikitext2_perplexity"] = float(wikitext2_score.group(1)) if wikitext2_score else None
    data["c4_perplexity"] = float(c4_score.group(1)) if c4_score else None
    data["average_accuracy"] = float(average_acc.group(1)) if average_acc else None

    # 解析每个任务的指标
    for match in task_pattern.finditer(log_text):
        task_name, version, n_shot, metric, direction, value, stderr = match.groups()
        tasks.append({
            "task": task_name.strip(),
            "version": int(version),
            "n_shot": int(n_shot),
            "metric": metric,
            "direction": direction.strip() if direction.strip() else None,
            "value": float(value),
            "stderr": float(stderr)
        })

    data["tasks"] = tasks
    return data



def parse_logs_with_automaton(logs_text):
    """使用自动机解析多个日志条目"""
    entries = []
    current_entry = []
    in_entry = False  # 状态标志：是否在一个日志条目中

    for line in logs_text.splitlines():
        # 检查是否进入一个新日志条目
        if line.startswith("Loading quantized model"):
            if in_entry and current_entry:
                # 处理前一个日志条目
                entries.append(parse_log_entry("\n".join(current_entry)))
                current_entry = []  # 重置当前日志条目
            in_entry = True

        # 如果在日志条目中，将行添加到当前条目
        if in_entry:
            current_entry.append(line)

        # 检查是否退出当前日志条目
        if line.startswith("Average Acc:"):
            in_entry = False
            if current_entry:
                # 处理当前日志条目并清空
                entries.append(parse_log_entry("\n".join(current_entry)))
                current_entry = []

    return entries

with open("exp.logs", "r") as f:
    logs_text = f.read()

# 使用自动机解析多个日志条目
parsed_logs = parse_logs_with_automaton(logs_text)

# 转换为 JSON 格式并输出
json_output = json.dumps(parsed_logs, indent=4)
# print(json_output)
def json_to_markdown_table(json_data):
    # 定义表头和分隔符
    headers = [
        "Method", "Wikitext2 Perplexity", "C4 Perplexity", "0-shot<br>avg",
        "Winogrande Accuracy", "Hellaswag Accuracy", 
        "ARC Challenge Accuracy", "ARC Easy Accuracy", "PIQA Accuracy"
    ]
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "| --- " * len(headers) + "|\n"

    # 遍历 JSON 数据列表并提取所需字段
    for entry in json_data:
        # 提取 method（quant_model_path 的倒数第二个路径项）
        quant_model_path = entry.get("quant_model_path", "").split("/")
        method = quant_model_path[-2] if len(quant_model_path) > 1 else "N/A"

        method = method.replace("-alpaca-4096", "")
        method = method.replace("w2gs128-fast-", "")
        method = method.replace("slide2", "window2")

        wikitext2_perplexity = entry.get("wikitext2_perplexity", "N/A")
        c4_perplexity = entry.get("c4_perplexity", "N/A")
        average_accuracy = entry.get("average_accuracy", "N/A")
        
        # 提取指定任务的 accuracy 值，默认 "N/A"
        task_accuracies = {
            "winogrande": "N/A",
            "hellaswag": "N/A",
            "arc_challenge": "N/A",
            "arc_easy": "N/A",
            "piqa": "N/A"
        }
        for task in entry.get("tasks", []):
            if task["task"] in task_accuracies:
                task_accuracies[task["task"]] = task["value"]

        # 将所有值转换为字符串并添加到表格行
        row = [
            method,
            str(wikitext2_perplexity), str(c4_perplexity), str(average_accuracy),
            str(task_accuracies["winogrande"]),
            str(task_accuracies["hellaswag"]),
            str(task_accuracies["arc_challenge"]),
            str(task_accuracies["arc_easy"]),
            str(task_accuracies["piqa"])
        ]
        markdown_table += "| " + " | ".join(row) + " |\n"

    return markdown_table

print(json_to_markdown_table(parsed_logs))