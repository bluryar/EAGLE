#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenO1-SFT数据集转换为ShareGPT格式的脚本
"""

import json
import argparse
import os
from typing import List, Dict, Any
from tqdm import tqdm


def convert_openo1_to_sharegpt(input_file: str, output_file: str) -> None:
    """
    将OpenO1-SFT数据集转换为ShareGPT格式
    
    Args:
        input_file: 输入的OpenO1-SFT JSONL文件路径
        output_file: 输出的ShareGPT格式JSONL文件路径
    """
    print(f"开始转换数据集: {input_file}")
    
    converted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"总共 {len(lines)} 条数据需要转换")
    
    for idx, line in enumerate(tqdm(lines, desc="转换进度")):
        if not line.strip():
            continue
            
        try:
            # 读取原始数据
            data = json.loads(line)
            
            # 转换为ShareGPT格式
            converted_item = {
                "id": f"openo1_sft_{idx}",
                "conversations": [
                    {
                        "from": "human",
                        "value": data["prompt"]
                    },
                    {
                        "from": "gpt",
                        "value": data["response"]
                    }
                ]
            }
            
            converted_data.append(converted_item)
            
        except json.JSONDecodeError as e:
            print(f"解析JSON失败，行 {idx + 1}: {e}")
            continue
        except KeyError as e:
            print(f"缺少必要字段，行 {idx + 1}: {e}")
            continue
    
    # 保存转换后的数据
    print(f"保存转换后的数据到: {output_file}")
    
    # 保存为JSONL格式
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 同时保存为JSON格式（可选）
    json_output_file = output_file.replace('.jsonl', '.json')
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！")
    print(f"- 总共转换了 {len(converted_data)} 条对话")
    print(f"- JSONL格式保存在: {output_file}")
    print(f"- JSON格式保存在: {json_output_file}")


def main():
    parser = argparse.ArgumentParser(description='将OpenO1-SFT数据集转换为ShareGPT格式')
    parser.add_argument('--input', type=str, required=True, help='输入的OpenO1-SFT JSONL文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出的ShareGPT格式JSONL文件路径')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误：输入文件不存在: {args.input}")
        return
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 执行转换
    convert_openo1_to_sharegpt(args.input, args.output)


if __name__ == "__main__":
    main()
