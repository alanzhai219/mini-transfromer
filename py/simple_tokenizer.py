import json
import re
from typing import List, Dict

def load_tokenizer_config(file_path: str) -> Dict:
    """加载 tokenizer.json 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def encode(text: str, vocab: Dict[str, int], merges: List[str] = None) -> List[int]:
    """编码：将文本转换为 token ID 序列"""
    # 应用 normalizer
    text = "▁" + text.replace(" ", "▁")  # Prepend ▁ and replace space with ▁

    # 预分词：按字符分割（由于 pre_tokenizer 为 null）
    tokens = list(text)

    # 应用 BPE 合并（简化版，基于词汇表中的常见子词）
    if merges:
        # 真实场景下需根据 merges 列表合并，此处模拟简单合并
        i = 0
        while i < len(tokens) - 1:
            pair = tokens[i] + tokens[i + 1]
            if pair in vocab:
                tokens[i] = pair
                tokens.pop(i + 1)
            else:
                i += 1
    else:
        # 没有 merges 列表，尝试词汇表中存在的子词
        result = []
        i = 0
        while i < len(tokens):
            found = False
            for j in range(len(tokens), i, -1):
                candidate = "".join(tokens[i:j])
                if candidate in vocab:
                    result.append(candidate)
                    i = j
                    found = True
                    break
            if not found:
                result.append(tokens[i])
                i += 1
        tokens = result

    # 映射到 token ID
    token_ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]

    # 应用 post_processor：添加 <s> 前缀
    token_ids = [1] + token_ids  # <s> ID = 1

    return token_ids

def decode(token_ids: List[int], vocab: Dict[str, int]) -> str:
    """解码：将 token ID 序列转换回文本"""
    # 创建反向词汇表
    id_to_token = {v: k for k, v in vocab.items()}

    # 移除 <s> 和 </s>
    token_ids = [id_ for id_ in token_ids if id_ not in [1, 2]]  # <s> ID = 1, </s> ID = 2

    # 转换为 token
    tokens = [id_to_token.get(id_, "<unk>") for id_ in token_ids]

    # 应用 decoder
    # 1. ByteFallback: 直接处理（词汇表已包含字节 token）
    # 2. Fuse: 合并 token
    text = "".join(tokens)
    # 3. Replace: 将 ▁ 替换为空格
    text = text.replace("▁", " ")
    # 4. Strip: 去除开头多余空格
    text = text.lstrip()

    return text

# 示例使用
if __name__ == "__main__":
    config = load_tokenizer_config("/mnt/disk/models/Llama-2-7b-chat-hf/tokenizer.json")
    vocab = config["model"]["vocab"]
    merges = config["model"].get("merges", [])  # 可能为空

    # 测试编码和解码
    text = "Hello world"
    encoded = encode(text, vocab, merges)
    print("Encoded IDs:", encoded)
    decoded = decode(encoded, vocab)
    print("Decoded text:", decoded)
