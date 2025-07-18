import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_path = "/data-mnt/data/downloaded_ckpts/Qwen2.5-0.5B-Instruct"
output_dir = "./qwen_lora_output"  # 训练保存的目录
lora_model_path = os.path.join(output_dir, "best_model")  # Trainer保存的最佳模型目录


def generate_response(model, tokenizer, query, max_new_tokens=512):
    # 生成模型响应
    messages = [
        {"role": "user", "content": query}
    ]

    # 使用与训练时相同的聊天模板
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # 训练时一致
    )

    inputs = tokenizer(input_text, return_tensors="pt")

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,  # 启用采样
            top_p=0.8,  # Top-p 采样
            temperature=0.7,  # 温度
            pad_token_id=tokenizer.eos_token_id,  # 确保pad_token_id设置正确
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def main_inference():
    print("正在加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 确保与训练时一致

    print("正在加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",  # 使用auto以便将模型加载到合适的设备
        trust_remote_code=True,
    )
    base_model.eval()  # 切换到评估模式

    print(f"正在加载LoRA适配器来自: {lora_model_path}...")
    model = PeftModel.from_pretrained(base_model, lora_model_path)

    # 合并模型
    print("正在合并LoRA权重...")
    model = model.merge_and_unload()
    print("LoRA权重合并完成。")

    # 测试生成
    print("\n--- 模型测试 ---")
    queries = [
        "当前的美国总统是谁？",
        "当前的全球石油价格是多少？"
    ]

    for i, query in enumerate(queries):
        print(f"\n用户 ({i + 1}): {query}")
        response = generate_response(model, tokenizer, query)
        print(f"模型 ({i + 1}): {response}")

    # 保存合并模型
    merged_model_save_path = os.path.join(output_dir, "merged_qwen_lora_model")
    print(f"\n正在保存合并后的模型到: {merged_model_save_path}")
    model.save_pretrained(merged_model_save_path)
    tokenizer.save_pretrained(merged_model_save_path)
    print("合并模型保存完成。")


if __name__ == "__main__":
    main_inference()