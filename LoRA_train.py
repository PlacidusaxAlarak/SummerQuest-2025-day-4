import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType

model_path="/data-mnt/data/downloaded_ckpts/Qwen2.5-0.5B-Instruct"
output_dir="./qwen_lora_output"
train_file="qwen_data_train.jsonl"
eval_file="qwen_data_test.jsonl"
lora_rank=16
lora_alpha=32
lora_target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
]
num_train_epochs=3
learning_rate=2e-4
per_device_eval_batch_size=2
gradient_accumulation_steps=8
seed=42
max_length=512

class QwenJsonDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.samples = []
        self.tokenizer = tokenizer
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item=self.samples[idx]
        messages=item["messages"]
        #生成prompt和response
        prompt=self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True
        )
        response=messages[-1]["content"]+self.tokenizer.eos_token#response后加eos_token

        #对prompt和response进行分词
        prompt_ids=self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids=self.tokenizer.encode(response, add_special_tokens=False)

        #拼接得到完整的input_ids
        inputs_ids=prompt_ids + response_ids
        #只保留response部分的标签
        labels=[-100]*len(prompt_ids) + response_ids

        if len(inputs_ids) > max_length:
            inputs_ids = inputs_ids[:max_length]
            labels = labels[:max_length]
        
        return{
            "input_ids": inputs_ids,  # 返回原始的 list
            "labels": labels,       
            "attention_mask": [1] * len(inputs_ids) 
        }

def print_trainable_parameters(model):
    """打印可训练参数的数量"""
    trainable_params=0
    all_param=0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"训练参数数量: {trainable_params}, 占总参数的比例: {trainable_params / all_param:.2%}")

def main():
    torch.manual_seed(seed)
    tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    
    model=AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"  # 避免 SDPA 警告
    )
    lora_config=LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model=get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    #准备数据集和Data Collator
    train_dataset=QwenJsonDataset(train_file, tokenizer)
    eval_dataset=QwenJsonDataset(eval_file, tokenizer)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,  # 传入模型是可选的，但有助于处理一些模型的特殊需求
        label_pad_token_id=-100,  # 这是最重要的参数，告诉collator用-100来填充labels
        pad_to_multiple_of=8  # 可选，为了硬件优化，将长度填充到8的倍数，可以提升训练速度
    )

    #配置训练参数
    training_args=TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        report_to="none",
        seed=seed,
        dataloader_num_workers=2,
        load_best_model_at_end=True,#加载最佳模型
        metric_for_best_model="eval_loss",#以评估损失为标准
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
    )

    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    trainer.save_model(os.path.join(output_dir, "best_model"))
    print(f"LoRA微调完成，模型已保存到: {output_dir}")

if __name__ == "__main__":
    main()