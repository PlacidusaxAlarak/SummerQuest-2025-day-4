# SummerQuest-2025-day-4

## 项目简介

本项目旨在实现一个基于大语言模型（LLM）的智能助手，支持搜索增强问答能力，并基于 LoRA 微调技术进行模型训练与推理。项目流程包括数据生成、预处理、LoRA 微调、模型合并和推理
---

## 目录结构与主要文件

```
.
├── Generate_Data.py      # 生成原始问答数据（含检索增强）
├── Standard_Data.py      # 数据标准化与训练/测试集划分
├── LoRA_train.py         # LoRA 微调训练脚本
├── Inference.py          # 推理脚本（合并权重后进行模型问答）
├── Search_Tool.py        # 搜索工具（模型辅助搜索）
├── question_with_search.txt      # 需检索的问题集（自备）
├── question_without_search.txt   # 普通问答问题集（自备）
├── qwen_data.json        # 生成的问答数据
├── qwen_data_train.jsonl # 格式化后的训练集
├── qwen_data_test.jsonl  # 格式化后的测试集
└── ...                   # 其他依赖文件或模型参数
```

---

## 工作流程

1. **数据生成**  
   `Generate_Data.py`  
   - 读取问题文件，模型生成初步回答。
   - 检测是否需调用搜索工具，自动生成增强回复。
   - 输出合成数据到 `qwen_data.json`。

2. **数据标准化与拆分**  
   `Standard_Data.py`  
   - 将原始数据格式化为 Qwen 兼容格式。
   - 按比例拆分训练/测试集，分别保存为 `.jsonl` 文件。

3. **LoRA 微调**  
   `LoRA_train.py`  
   - 加载预训练模型与分词器。
   - 配置 LoRA 参数与目标模块。
   - 构建自定义 Dataset，启动 Trainer 进行微调。
   - 最优权重保存至 `qwen_lora_output/best_model`。

4. **推理与合并权重**  
   `Inference.py`  
   - 加载微调后模型和原始模型。
   - 合并 LoRA 权重，保存为最终模型。
   - 支持自定义问题推理与问答。

5. **搜索工具辅助**  
   `Search_Tool.py`  
   - 封装检索功能，模型可根据关键词生成相关内容。
   - 支持独立测试与模型集成调用。

---

## 快速开始

### 环境准备

- Python 3.8+
- 推荐使用 GPU（支持 bfloat16），若无则自动切换为 float32
- 依赖库：`transformers`, `peft`, `torch`, `tqdm`

```bash
pip install torch transformers peft tqdm
```

### 1. 生成问答数据

需要准备两份问题集：

- `question_with_search.txt`：需要检索的问题
- `question_without_search.txt`：无需检索的问题

执行：

```bash
python Generate_Data.py
```

生成 `qwen_data.json`

---

### 2. 数据标准化与拆分

```bash
python Standard_Data.py
```

生成 `qwen_data_train.jsonl` 和 `qwen_data_test.jsonl`

---

### 3. LoRA 微调训练

```bash
python LoRA_train.py
```

训练输出在 `./qwen_lora_output/`

---

### 4. 推理与权重合并

```bash
python Inference.py
```

合并后的模型保存于 `./qwen_lora_output/merged_qwen_lora_model/`

---

## 主要参数说明

- **模型路径** 可根据实际情况修改（如 Qwen、DeepSeek 等）。
- **LoRA配置** 可调整 `lora_rank`, `lora_alpha`, `target_modules` 等参数以适配不同模型。
- **数据文件** 建议格式为一行一个问题或一条 JSON，详见代码注释部分。

---
