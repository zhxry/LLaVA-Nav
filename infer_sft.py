import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel


if __name__ == "__main__":
    base_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    adapter_path = "model/sft-qwen7b/"
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = model.to("cuda")
    model.eval()

    image_path = "data/Train/10/10-1.jpg"
    image = Image.open(image_path).convert("RGB").resize((512, 512), Image.BICUBIC)
    # prompt = (
    #     "This is a 512*512 remote sensing image for uav navigation. "
    #     + "The red circle marks the starting point, the yellow circle marks the destination, "
    #     + "blue dots (or rectangles) indicate must-pass waypoints (or regions), "
    #     + "and green dots (or rectangles) represent no-fly points (or regions). "
    #     + "Please provide a valid and feasible path in the format: "
    #     + "[(x1, y1), (x2, y2), ..., (xn, yn)]."
    # )
    # prompt = (
    #     "这是一张 5128*5128 的遥感图像。你需要为无人机规划一条可行的飞行路径。"
    #     + "图中红色圆为起点，黄色圆为终点，紫色点为必经点，绿色为禁飞区，请给出一条像素坐标表示的可行路径。"
    # )
    prompt = (
        "请描述这张遥感图像。"
    )

    messages = [
        {
            "content": [
                {"index": None, "type": "text", "text": prompt},
                {"index": 0, "type": "image", "text": None}
            ],
            "role": "user"
        }
    ]

    with torch.no_grad():  # 节省显存，加速推理
        text = processor.apply_chat_template(messages, tokenize=False)
        inputs = processor(text=text, images=image, return_tensors="pt").to("cuda", torch.bfloat16)

        # 将输入移到GPU
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        # print(inputs.keys())
        # print(inputs)
        # print("model device:", model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            # temperature=0.7,
            # top_p=0.9,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        response = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"Model response: {response}")
