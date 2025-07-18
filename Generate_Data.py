import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random
import os
import re

from Search_Tool import SearchTool as search_tool


class Data:
    def __init__(self, model_path="/data-mnt/data/downloaded_ckpts/DeepSeek-R1-Distill-Qwen-7B"):
        print(f"正在从{model_path}加载模型")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = self.model.device

        self.search_tool_format = self.get_search_tool_format()
        self.search_tool = search_tool(self.model, self.tokenizer)
        print("模型和工具初始化完成")

    def get_search_tool_format(self):
        # 返回搜索工具的JSON格式
        return {
            "type": "function",
            "function": {
                "name": "search",
                "description": "搜索引擎，在需要获取实时信息、最新数据、具体事实查询时需要调用此工具",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "使用搜索引擎所需的关键词"},
                        "top_k": {"type": "number", "default": 3, "description": "返回的搜索结果数量"}
                    },
                    "required": ["keyword"]
                }
            }
        }

    def _generate(self, messages: list, max_tokens: int) -> str:
        # 私有的生成函数
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def generate_initial_response(self, question: str, use_search_prompt: bool) -> str:
        # 模型的初步回复
        if use_search_prompt:
            system_prompt = (
                f"你是一个智能助手，可以使用搜索工具获取信息。当需要实时信息、最新数据或具体事实时，请使用搜索工具。可用工具：\n{json.dumps(self.search_tool_format, ensure_ascii=False, indent=2)}"
            )
            max_tokens = 1024
        else:
            system_prompt = "你是一个智能助手，请仔细思考用户的问题并给出详细回答。"
            max_tokens = 2048

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        return self._generate(messages, max_tokens=max_tokens)

    def extract_search_keyword(self, response_text: str, question: str) -> str | None:
        # 对提取关键词进行了优化
        # 匹配多种精确的、类似代码的调用格式
        patterns = [
            r'["“]\s*(?:keyword|关键词)\s*["”]\s*[:=]\s*["“]([^"”]+)["”]',  # 键值对keyword:,关键词:
            r'\bsearch\s*\(\s*["“]([^"”]+)["”]',  # search()
            r'\bsearch\s+keyword\s*=\s*["“]([^"”]+)["”]',  # search(keyword="")
            r'\{\s*(?:keyword|关键词)\s*:\s*["“]([^"”]+)["”]',  # {keyword:} or {关键词:}
        ]

        for pattern in patterns:
            match = re.search(pattern, response_text)
            if match:
                return match.group(1).strip()

        # 策略:匹配指令式的自然语言调用
        intent_pattern = r'(?:调用|使用|输入|执行)\s*(?:搜索工具|搜索)\s*.*?关键词\s*(?:是|为)?\s*[:：]?\s*["“]([^"”]+)["”]'
        match = re.search(intent_pattern, response_text)
        if match:
            return match.group(1).strip()

        # 策略:分析 <think> 块中的模糊意图 (作为强大的后备)
        think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
        if think_match:
            thought = think_match.group(1)
            if any(word in thought for word in ["搜索", "search", "调用", "查找", "tool"]):
                keyword_extract_match = re.search(r'关键词[是为]?\s*[“"]([^"”]+)[”"]', thought)
                if keyword_extract_match:
                    return keyword_extract_match.group(1).strip()
                else:
                    # 使用整个问题作为关键词
                    return question
        # 如果所有策略都失败，则返回 None
        return None

    def execute_search(self, keyword: str, question: str) -> str:
        # 给出最终response
        search_results = self.search_tool.search(keyword, top_k=3)
        search_result_text = "\n".join(
            [f"搜索结果{i + 1}: {result}" for i, result in enumerate(search_results)]
        )
        final_messages = [
            {"role": "system", "content": "你是一个智能助手，请基于搜索结果回答用户问题"},
            {"role": "user", "content": f"问题: {question}\n\n搜索结果:\n{search_result_text}"}
        ]
        return self._generate(final_messages, max_tokens=2048)

    def generate_data(self, with_search_file: str, without_search_file: str, output_file: str, num_samples: int):

        def load_questions(file_path):
            if not os.path.exists(file_path):
                print(f"文件 {file_path} 不存在，将跳过。")
                return []
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]

        questions_with_search = load_questions(with_search_file)[:num_samples]
        questions_without_search = load_questions(without_search_file)[:num_samples]

        synthesized_data = []

        print("处理需要搜索的问题...")
        for question in tqdm(questions_with_search):  # 进度条
            try:
                initial_response = self.generate_initial_response(question, use_search_prompt=True)
                keyword = self.extract_search_keyword(initial_response, question)
                if keyword:
                    final_response = self.execute_search(keyword, question)
                    synthesized_data.append({"question": question, "response": final_response, "use_search": True})
                else:
                    synthesized_data.append({"question": question, "response": initial_response, "use_search": False})
            except Exception as e:
                print(f"处理问题时出错: {question}, 错误: {e}")

        print("处理不需要搜索的问题...")
        for question in tqdm(questions_without_search):
            try:
                response = self.generate_initial_response(question, use_search_prompt=False)
                synthesized_data.append({"question": question, "response": response, "use_search": False})
            except Exception as e:
                print(f"处理问题时出错: {question}, 错误: {e}")

        random.shuffle(synthesized_data)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(synthesized_data, f, ensure_ascii=False, indent=2)
        print(f"\n合成数据完成! 共生成{len(synthesized_data)}条数据, 保存到{output_file}")
        return synthesized_data


if __name__ == "__main__":
    synthesizer = Data()
    synthesizer.generate_data(
        with_search_file="question_with_search.txt",
        without_search_file="question_without_search.txt",
        output_file="qwen_data.json",
        num_samples=50
    )