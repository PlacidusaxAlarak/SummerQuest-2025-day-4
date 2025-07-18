import json
import random
from typing import List, Dict, Tuple

Search_Tool_Definition = {
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
System_Prompt_Without_Tool = "你是一个智能助手,请仔细思考用户的问题并给出详细回答"
System_Prompt_With_Tool = f"""你是一个智能助手，可以使用搜索工具获取信息。当需要实时信息、最新数据或具体事实时，请使用搜索工具。可用工具：
{json.dumps(Search_Tool_Definition, ensure_ascii=False, indent=2)}
请仔细思考用户的问题，判断是否需要使用搜索工具。"""


class DataFormatter:
    # 数据格式化
    def format_for_qwen(self, data: List[Dict]) -> List[Dict]:
        return [self._format_single_item(item) for item in data]

    def _format_single_item(self, item: Dict) -> Dict:
        # 格式化每个数据项
        use_search = item.get("use_search", False)
        system_content = System_Prompt_With_Tool if use_search else System_Prompt_Without_Tool
        conversation = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["response"]}
        ]
        return {
            "messages": conversation,
            "use_search": use_search
        }

    def split_train_test(self, data: List[Dict], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        # 划分训练集和测试集
        random.shuffle(data)
        split_index = int(len(data) * train_ratio)
        return data[:split_index], data[split_index:]

    def save_jsonl(self, data: List[Dict], filename: str):
        # 将数据保存为jsonl文件
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def process_data(self, input_file: str, output_file: str = "qwen_data"):
        max_train_samples = 300
        max_test_samples = 85
        total_samples = max_train_samples + max_test_samples

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            print(f"加载了{len(raw_data)}条数据")
        except FileNotFoundError:
            print(f"文件 {input_file} 不存在")
            return [], []
        # 格式化
        formatted_data = self.format_for_qwen(raw_data)
        if len(formatted_data) > total_samples:
            formatted_data = formatted_data[:total_samples]

        # 划分训练集和测试集
        train_data, test_data = self.split_train_test(formatted_data, train_ratio=0.8)
        # 保证不超过最大限制
        train_data = train_data[:max_train_samples]
        test_data = test_data[:max_test_samples]
        self.save_jsonl(train_data, f"{output_file}_train.jsonl")
        self.save_jsonl(test_data, f"{output_file}_test.jsonl")

        print(f"训练集:{len(train_data)}条数据->{output_file}_train.jsonl")
        print(f"测试集:{len(test_data)}条数据->{output_file}_test.jsonl")

        train_with_search = sum(1 for item in train_data if item.get("use_search"))
        test_with_search = sum(1 for item in test_data if item.get("use_search"))
        print(f"训练集使用搜索工具的数据条数: {train_with_search}")
        print(f"测试集使用搜索工具的数据条数: {test_with_search}")

        return train_data, test_data


if __name__ == "__main__":
    formatter = DataFormatter()
    train_data, test_data = formatter.process_data("qwen_data.json", output_file="qwen_data")
    print(f"处理完成！训练集: {len(train_data)}条，测试集: {len(test_data)}条")