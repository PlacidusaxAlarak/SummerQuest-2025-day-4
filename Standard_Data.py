import json
import os
import glob # 导入 glob 模块用于查找文件
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

#设置路径
MODEL_PATH = "/data-mnt/data/camp-2025/SummerQuest-2025/submission/申益泽/day-5/models/Qwen2.5-Math-7B"
DATASET_PATH = "/data-mnt/data/camp-2025/SummerQuest-2025/submission/申益泽/day-5/dataset/gsm8k"
DATASET_SPLIT = "train" #选择'train'或'test'

OUTPUT_FILE = f"gsm8k_{DATASET_SPLIT}_distill_data.jsonl" #
MAX_TOKENS = 2048
TEMPERATURE = 0.0#数学题要用精准的输出

print(f"从本地路径 '{MODEL_PATH}' 加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print(f"从本地路径 '{MODEL_PATH}' 初始化 VLLM...")
llm = LLM(model=MODEL_PATH, trust_remote_code=True, tensor_parallel_size=2)#使用两个GPU进行

# --- Prompt 定义不变 ---
def create_prompt(question):
    messages = [
        {"role": "system", "content": "你是一位顶级的数学解题专家，请用清晰、分步的思路解决问题。"},
        {"role": "user", "content": f"请解决以下数学问题。请一步一步地思考，并给出详细的解题过程。最后，将最终答案放在 \\boxed{{}} 中。\n\n问题：{question}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_p=1.0,
    max_tokens=MAX_TOKENS,
    stop_token_ids=[tokenizer.eos_token_id]
)

#加载数据集的文件(parquet)
print(f"正在从本地路径 '{DATASET_PATH}' 为 '{DATASET_SPLIT}' 分割查找 Parquet 文件...")

# 构造目标分割的子目录路径，例如 ".../gsm8k/train"
split_directory = os.path.join(DATASET_PATH, DATASET_SPLIT)
if not os.path.isdir(split_directory):
    raise FileNotFoundError(f"错误：找不到数据集分割目录 '{split_directory}'。请检查 DATASET_PATH 和 DATASET_SPLIT 是否正确。")

parquet_files = glob.glob(os.path.join(split_directory, '*.parquet'))

if not parquet_files:
    raise FileNotFoundError(f"错误：在目录 '{split_directory}' 中没有找到任何 .parquet 文件。")

print(f"找到 {len(parquet_files)} 个 Parquet 文件: {parquet_files}")
print(f"正在加载数据集...")


dataset = load_dataset("parquet", data_files={DATASET_SPLIT: parquet_files})[DATASET_SPLIT]#parquet告诉Huggingface,不是HF中的数据集，返回一个字典
print("数据集加载成功！")
print(dataset)



original_questions = [sample['question'] for sample in dataset]
prompts = [create_prompt(q) for q in tqdm(original_questions, desc="构建 Prompts")]

print(f"开始使用 VLLM 对 {len(prompts)} 条数据进行批量生成...")
outputs = llm.generate(prompts, sampling_params)
print("生成完成！")

# ... 保存文件的代码不变 ...
print(f"正在将结果保存到 {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for i, output in enumerate(tqdm(outputs, desc="保存文件中")):
        data_record = {
            "instruction": original_questions[i],
            "output": output.outputs[0].text.strip()
        }
        f.write(json.dumps(data_record, ensure_ascii=False) + '\n')

print(f"阶段一完成！结果已保存至 {OUTPUT_FILE}")