import torch
import sys
from typing import List, Dict

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("未找到Transformer库")
    sys.exit(1)


class SearchTool:
    def __init__(self, model, tokenizer):

        if model is None or tokenizer is None:
            raise ValueError("请加载正确的模型和分词器实例")

        self.model = model
        self.tokenizer = tokenizer
        # 确保有pad_token, 避免警告
        if self.tokenizer.pad_token is None:
            self.tokenizer_pad_token = self.tokenizer.eos_token

        self.device = model.device
        print(f"SearchTool已经被初始化,模型在设备{self.device}上")

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        # 一个私有方法,负责调用模型生成文本
        try:
            # 准备输入
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True  # 加上能触发模型回复的token
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # 解码，并且返回结果
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return response
        except Exception as e:
            print(f"错误信息:{e}")
            return ""

    def search(self, keyword: str, top_k: int = 4) -> str:
        """
        根据关键词进行搜索，并且返回结果列表
        """
        prompt_messages = {
            "role": "user",
            "content": (
                f"你现在是一个搜索引擎助手,对于任何的输入信息,请你给出{min(top_k, 5)}个合理的，最相近的搜索结果."
                "以列表的方式呈现,每个结果占一行.\n\n"
                f"输入信息：{keyword}"
            )
        }
        # 调用模型生成内容
        generated_content = self._generate([prompt_messages])

        # 如果失败,返回备用结果
        if not generated_content:
            return self._get_backup_result(keyword, top_k)

        # 解析模型输出
        try:
            lines = generated_content.strip().split("</think>")[-1].strip().split('\n')
            # 清理空行并且获取前top_k个结果
            results = [line.strip() for line in lines if line.strip()]
            return results[:top_k]
        except Exception:
            # 如果解析出错,返回备用结果
            return self._get_backup_result(keyword, top_k)

    def _get_backup_result(self, keyword: str, top_k: int) -> List[str]:
        """当主流程失败时，提供一组硬编码的备用结果。"""
        print(f"主流程失败，使用备用结果")
        backup_data = [
            f"关于'{keyword}'的搜索结果1：这是一个模拟的搜索结果。",
            f"关于'{keyword}'的搜索结果2：这是另一个模拟的搜索结果。",
            f"关于'{keyword}'的搜索结果3：这是第三个模拟的搜索结果。"
        ]
        return backup_data[:top_k]


if __name__ == "__main__":
    print("正在测试SearchTool\n")
    model_path_for_test = "/data-mnt/data/downloaded_ckpts/DeepSeek-R1-Distill-Qwen-7B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path_for_test)
        model = AutoModelForCausalLM.from_pretrained(
            model_path_for_test,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        search_tool = SearchTool(model, tokenizer)
        keyword_to_search = " ".join(sys.argv[1:])  # 跟在命令行后面的参数

        print(f"正在搜索:{keyword_to_search}")
        results = search_tool.search(keyword_to_search, top_k=3)
        print(f"——————搜索结果——————\n")
        for i, result in enumerate(results, 1):
            print(f"{result}\n")
    except Exception as e:
        print(f"测试失败: {e}")
        sys.exit(1)
