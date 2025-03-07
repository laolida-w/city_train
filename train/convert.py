import json
import os

# 读取原始数据
input_path = "qwen_outputs_test.json"
output_dir = "output_llamafactory"
os.makedirs(output_dir, exist_ok=True)

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# LLaMA-Factory 格式数据
llama_data = []

for item in data:
    image_id = item["image_id"]
    image_path = f"images/{image_id}.jpg"  # 假设图片存放在 images 目录

    # 任务 1: image_caption 任务
    if "image_description" in item:
        llama_data.append({
            "instruction": "[image_caption] Describe the content of the image.",
            "input": image_path,
            "output": f"<image> {item['image_description']}"
        })

    # 任务 2: QA 任务
    if "question_answer" in item:
        for qa in item["question_answer"]:
            llama_data.append({
                "instruction": f"[QA] {qa['question']}",
                "input": image_path,
                "output": f"<image>{qa['answer']}"
            })

    # 任务 3: region_caption 任务
    if "bboxes" in item:
        for bbox in item["bboxes"]:
            llama_data.append({
                "instruction": f"[region_caption] Describe the specified region in the image.(Region: {bbox['bbox']})",
                "input": f"{image_path}",
                "output": f"<image>{bbox['region_description']}"
            })

    # 任务 4: grounded_caption 任务
    if "grounded_caption" in item:
        llama_data.append({
            "instruction": "[grounded_caption] Provide a detailed caption with detected regions.",
            "input": image_path,
            "output": f"<image>{item['grounded_caption']}"
        })

# 保存转换后的数据
output_path = os.path.join(output_dir, "llamafactory_task1_dataset.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(llama_data, f, indent=4, ensure_ascii=False)

print(f"转换完成，数据保存在 {output_path}")


