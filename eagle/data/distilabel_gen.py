# -*- coding: utf-8 -*-
import json
import os
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm.auto import tqdm

try:
    from distilabel.pipeline import Pipeline
    from distilabel.steps import LoadDataFromDicts
    from distilabel.steps.tasks import TextGeneration
    from distilabel.llms import OpenAILLM
    from datasets import Dataset, concatenate_datasets
    import pyarrow.parquet as pq
except ImportError as e:
    print(f"导入错误: {e}")
    print("请先安装依赖项: pip install -r requirements.txt")
    exit(1)


def load_ultra_dataset(directory_path: str) -> List[Dict]:
    """加载Ultra数据集"""
    parquet_files = glob.glob(os.path.join(directory_path, "*.parquet"))
    result_list = []
    
    for file_path in parquet_files:
        table = pq.read_table(file_path)
        records = table.to_pydict()
        num_rows = len(next(iter(records.values())))
        row_list = [
            {key: records[key][i] for key in records} 
            for i in range(num_rows)
        ]
        result_list.extend(row_list)
    
    return result_list


def load_sharegpt_dataset(dataset_path: str) -> List[Dict]:
    """加载ShareGPT数据集"""
    data = []
    if dataset_path.endswith('.jsonl'):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    elif dataset_path.endswith('.json'):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        raise ValueError(f"不支持的文件格式: {dataset_path}")
    
    return data


def prepare_conversation_data(dataset_path: str, dataset_type: str = "sharegpt", max_examples: Optional[int] = None) -> List[Dict]:
    """准备对话数据，转换为 distilabel 可以处理的格式"""
    # 加载数据集
    if dataset_type == "ultra":
        dataset = load_ultra_dataset(dataset_path)
    else:
        dataset = load_sharegpt_dataset(dataset_path)
    
    # 如果指定了最大示例数，则限制数据量
    if max_examples is not None and max_examples > 0:
        original_length = len(dataset)
        dataset = dataset[:max_examples]
        print(f"调试模式：使用前 {max_examples} 条数据（原始数据量: {original_length}）")
    
    # 准备输出数据 - 为每个助手回合创建一个生成任务
    output_data = []
    
    for idx, conversation in enumerate(dataset):
        conversations_list = conversation.get("conversations", [])
        
        # 处理不同的数据格式
        if not conversations_list:
            if "items" in conversation:
                conversations_list = conversation["items"]
            else:
                conversations_list = conversation.get("messages", [])
        
        conversation_id = conversation.get("id", f"conversation_{idx}")
        
        # 为每个助手回合创建生成任务
        context = ""
        assistant_turn_count = 0
        
        for i, msg in enumerate(conversations_list):
            role = msg.get("from") or msg.get("role")
            content = msg.get("value") or msg.get("content")
            
            if role in ["human", "user"]:
                context += f"Human: {content}\n\n"
            elif role in ["gpt", "assistant"]:
                # 为当前助手回合创建生成任务
                current_context = context + "Assistant:"
                
                output_data.append({
                    "conversation_context": current_context,
                    "conversation_id": conversation_id,
                    "original_conversations": conversations_list,
                    "assistant_turn_index": assistant_turn_count,
                    "original_response": content  # 保存原始回答用于比较
                })
                
                # 暂时用占位符更新上下文（后续会被替换）
                context += f"Assistant: [PLACEHOLDER_{assistant_turn_count}]\n\n"
                assistant_turn_count += 1
    
    return output_data


def reconstruct_conversations(processed_data: List[Dict]) -> List[Dict]:
    """重建完整的对话"""
    # 按 conversation_id 分组生成的回答
    responses_by_conversation = {}
    
    for item in processed_data:
        conv_id = item["conversation_id"]
        if conv_id not in responses_by_conversation:
            responses_by_conversation[conv_id] = {
                "original_conversations": item["original_conversations"],
                "responses": {}
            }
        
        turn_index = item["assistant_turn_index"]
        generated_response = item["generation"]  # distilabel 生成的回答
        responses_by_conversation[conv_id]["responses"][turn_index] = generated_response
    
    # 重建对话
    output_data = []
    
    for conv_id, conv_data in responses_by_conversation.items():
        original_conversations = conv_data["original_conversations"]
        generated_responses = conv_data["responses"]
        
        # 重建对话
        new_conversations = []
        assistant_turn_count = 0
        
        for msg in original_conversations:
            role = msg.get("from") or msg.get("role")
            content = msg.get("value") or msg.get("content")
            
            if role in ["human", "user"]:
                new_conversations.append({
                    "from": "human",
                    "value": content
                })
            elif role in ["gpt", "assistant"]:
                # 使用生成的回答替换原始回答
                new_content = generated_responses.get(assistant_turn_count, content)
                new_conversations.append({
                    "from": "gpt",
                    "value": new_content
                })
                assistant_turn_count += 1
        
        output_data.append({
            "id": conv_id,
            "conversations": new_conversations
        })
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description='使用distilabel重新生成ShareGPT数据集的回答')
    parser.add_argument('--dataset', type=str, required=True, help='ShareGPT数据集文件路径')
    parser.add_argument('--vllm_url', type=str, required=True, help='vLLM服务的API地址')
    parser.add_argument('--output_dir', type=str, default='./regenerated_data', help='输出目录')
    parser.add_argument('--dataset_type', type=str, default='sharegpt', 
                        choices=['sharegpt', 'ultra'], help='数据集类型')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成文本的温度参数')
    parser.add_argument('--max_new_tokens', type=int, default=2048, help='每次生成的最大token数')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-72B-Instruct', 
                        help='模型名称')
    parser.add_argument('--batch_size', type=int, default=4, help='批处理大小')
    parser.add_argument('--use_cache', action='store_true', help='是否使用缓存')
    parser.add_argument('--timeout', type=float, default=120.0, help='API请求超时时间(秒)')
    parser.add_argument('--max_retries', type=int, default=3, help='最大重试次数')
    parser.add_argument('--max_examples', type=int, default=None, help='用于调试：仅使用前N条数据')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 准备数据
    print(f"开始加载数据集: {args.dataset}")
    input_data = prepare_conversation_data(args.dataset, args.dataset_type, args.max_examples)
    print(f"准备了 {len(input_data)} 个生成任务")
    
    # 如果是调试模式，提供一些额外信息
    if args.max_examples is not None:
        print(f"🔧 调试模式：仅处理前 {args.max_examples} 条对话数据")
        if len(input_data) > 0:
            print(f"📊 示例数据预览：")
            sample_data = input_data[0]
            print(f"  - 对话ID: {sample_data.get('conversation_id', 'N/A')}")
            print(f"  - 助手回合索引: {sample_data.get('assistant_turn_index', 'N/A')}")
            print(f"  - 上下文长度: {len(sample_data.get('conversation_context', ''))}")
            print(f"  - 原始回答长度: {len(sample_data.get('original_response', ''))}")
            print(f"  - 上下文预览: {sample_data.get('conversation_context', '')[:100]}...")
        else:
            print("⚠️ 没有找到任何对话数据")
    
    # 创建流水线
    with Pipeline(name="sharegpt-regeneration") as pipeline:
        # 第一步：加载数据
        load_data = LoadDataFromDicts(
            name="load_data",
            data=input_data,
            batch_size=args.batch_size
        )
        
        # 第二步：初始化LLM
        llm = OpenAILLM(
            base_url=args.vllm_url,
            api_key="sk-none",  # type: ignore
            model=args.model_name,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        
        # 第三步：重新生成助手回答
        response_generator = TextGeneration(
            name="response_generator",
            llm=llm,
            input_batch_size=args.batch_size,
            system_prompt=(
                "你是一个有用的AI助手。请根据对话上下文生成合适的回答。"
                "回答应该自然、有帮助，并与对话的语气和风格保持一致。"
            ),
            template="{{ conversation_context }}",
            columns=["conversation_context"]
        )
        
        # 连接步骤
        load_data >> response_generator
    
    # 运行流水线
    print(f"开始处理数据集")
    print(f"使用模型: {args.model_name}")
    print(f"vLLM URL: {args.vllm_url}")
    print(f"批处理大小: {args.batch_size}")
    print(f"温度: {args.temperature}")
    print(f"最大新生成tokens: {args.max_new_tokens}")
    if args.max_examples is not None:
        print(f"调试模式：最大示例数: {args.max_examples}")
    
    # 测试 API 连接
    try:
        import requests
        test_url = f"{args.vllm_url.rstrip('/')}/models"
        print(f"测试 API 连接: {test_url}")
        response = requests.get(test_url, timeout=10)
        print(f"API 连接测试结果: {response.status_code}")
        if response.status_code == 200:
            print("API 连接正常")
        else:
            print(f"API 连接异常: {response.text}")
    except Exception as e:
        print(f"API 连接测试失败: {e}")
        print("请检查 vLLM 服务是否正常运行")
    
    try:
        distiset = pipeline.run(
            use_cache=args.use_cache,
            parameters={
                "response_generator": {
                    "llm": {
                        "generation_kwargs": {
                            "temperature": args.temperature,
                            "max_new_tokens": args.max_new_tokens,
                            "top_p": 0.9,
                        }
                    }
                }
            }
        )
    except Exception as e:
        print(f"运行流水线时出错: {e}")
        print("可能的原因:")
        print("1. vLLM 服务未启动或配置错误")
        print("2. 模型名称不正确")
        print("3. API 端点路径不正确")
        print("4. 网络连接问题")
        raise
    
    # 保存结果
    output_path = Path(args.output_dir)
    
    # 保存为Arrow格式
    distiset.save_to_disk(
        distiset_path=output_path / "distiset",
        max_shard_size="500MB",
        num_proc=4,
        save_card=True,
        save_pipeline_config=True,
        save_pipeline_log=True
    )
    
    # 获取生成的数据
    config_name = "default" if "default" in distiset else next(iter(distiset.keys()))
    dataset = distiset[config_name]
    
    # 处理DatasetDict结构
    if hasattr(dataset, 'keys'):
        combined_dataset = concatenate_datasets([dataset[split] for split in dataset.keys()])
    else:
        combined_dataset = dataset
    
    # 转换为列表格式以便处理
    processed_data = []
    for item in combined_dataset:
        processed_data.append(item)
    
    # 重建完整对话
    print("重建完整对话...")
    sharegpt_data = reconstruct_conversations(processed_data)
    
    # 保存为JSON
    json_path = output_path / "regenerated_complete.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
    
    # 保存为JSONL
    jsonl_path = output_path / "regenerated_complete.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in sharegpt_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"处理完成！")
    print(f"总共处理了 {len(sharegpt_data)} 条对话")
    print(f"结果已保存到: {output_path}")
    print(f"JSON格式: {json_path}")
    print(f"JSONL格式: {jsonl_path}")
    
    # 显示示例
    if sharegpt_data:
        print("\n示例数据：")
        print(json.dumps(sharegpt_data[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main() 

# 使用示例:
# python distilabel_gen.py \
#   --dataset /path/to/dataset.jsonl \
#   --vllm_url http://localhost:8000/v1 \
#   --model_name Qwen/Qwen2.5-72B-Instruct \
#   --batch_size 4 \
#   --temperature 0.7 \
#   --max_new_tokens 2048 \
#   --output_dir ./output \
#   --use_cache
#
# 调试模式示例 (只使用前5条数据):
# python distilabel_gen.py \
#   --dataset /path/to/dataset.jsonl \
#   --vllm_url http://localhost:8000/v1 \
#   --model_name Qwen/Qwen2.5-72B-Instruct \
#   --batch_size 2 \
#   --max_examples 5 \
#   --temperature 0.7 \
#   --output_dir ./debug_output