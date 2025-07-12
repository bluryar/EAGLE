# -*- coding: utf-8 -*-
import json
import requests
import time
import random
import argparse
import os
import threading
import queue
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import pyarrow.parquet as pq
import os
import glob
class ShareGPTReGenerator:
    """用于重新生成ShareGPT数据集回答的类"""

    def __init__(self, vllm_url: str, output_dir: str, 
                 temperature: float = 0.7, max_tokens: int = 2048,
                 num_threads: int = 4, request_timeout: int = 60):
        """
        初始化ShareGPT数据重新生成器

        Args:
            vllm_url: vLLM服务的API地址
            output_dir: 输出数据集的目录
            temperature: 生成文本的温度参数
            max_tokens: 每次生成的最大token数
            num_threads: 并行处理的线程数
            request_timeout: API请求超时时间(秒)
        """
        self.vllm_url = vllm_url
        self.output_dir = output_dir
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_threads = num_threads
        self.request_timeout = request_timeout

        # 用于线程安全的操作
        self.result_lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.api_rate_limit_lock = threading.Lock()
        self.last_api_call = 0
        
        # 动态调整的API调用间隔
        self.min_api_interval = 0.5  # 增加基础间隔
        self.dynamic_interval = 0.5  # 动态调整的间隔
        
        # 重试和背压控制
        self.max_retries = 3  # 最大重试次数
        self.consecutive_failures = 0  # 连续失败次数
        self.failure_threshold = 5  # 失败阈值，超过此值增加延迟
        self.success_count = 0  # 成功计数
        self.failure_count = 0  # 失败计数

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    def load_ultra_dataset(self, directory_path:str) -> List[Dict]:
        # 获取指定目录下所有parquet文件路径
        parquet_files = glob.glob(os.path.join(directory_path, "*.parquet"))

        result_list = []

        # 读取每个parquet文件并添加到列表中
        for file_path in parquet_files:
            table = pq.read_table(file_path)
            # 将Table转换为字典列表
            records = table.to_pydict()

            # 转换为记录列表
            num_rows = len(next(iter(records.values())))
            row_list = [
                {key: records[key][i] for key in records} 
                for i in range(num_rows)
            ]

            result_list.extend(row_list)

        return result_list

    def load_sharegpt_dataset(self, dataset_path: str) -> List[Dict]:
        """
        加载ShareGPT数据集

        Args:
            dataset_path: ShareGPT数据集文件路径，格式为jsonl或json

        Returns:
            加载的数据集
        """
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

    def call_vllm_api(self, prompt: str) -> str:
        """
        调用vLLM API生成回答，带重试机制和动态背压控制

        Args:
            prompt: 输入提示

        Returns:
            生成的回答
        """
        for attempt in range(self.max_retries):
            try:
                # 动态调整API调用间隔
                with self.api_rate_limit_lock:
                    current_time = time.time()
                    time_since_last_call = current_time - self.last_api_call
                    
                    # 根据失败率动态调整间隔
                    if self.consecutive_failures > self.failure_threshold:
                        self.dynamic_interval = min(self.dynamic_interval * 1.5, 5.0)  # 最大5秒
                    elif self.consecutive_failures == 0 and self.success_count > 10:
                        self.dynamic_interval = max(self.dynamic_interval * 0.9, self.min_api_interval)  # 逐渐减少
                    
                    sleep_time = max(self.dynamic_interval - time_since_last_call, 0)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                    self.last_api_call = time.time()

                payload = {
                    "model": "/root/Qwen3-4B-FP8",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stop": []
                }

                # 根据尝试次数调整超时时间
                timeout = self.request_timeout + (attempt * 30)  # 每次重试增加30秒
                
                response = requests.post(self.vllm_url, json=payload, timeout=timeout)
                response.raise_for_status()
                result = response.json()
                
                # 成功时重置失败计数
                with self.api_rate_limit_lock:
                    self.consecutive_failures = 0
                    self.success_count += 1
                
                return result["choices"][0]["message"]["content"]

            except requests.exceptions.ReadTimeout as e:
                print(f"API调用超时 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                with self.api_rate_limit_lock:
                    self.consecutive_failures += 1
                    self.failure_count += 1
                
                if attempt < self.max_retries - 1:
                    # 指数退避重试
                    sleep_time = (2 ** attempt) * random.uniform(1, 2)
                    print(f"等待 {sleep_time:.2f} 秒后重试...")
                    time.sleep(sleep_time)
                else:
                    print(f"最终超时失败: {e}")
                    return ""
                    
            except requests.exceptions.ConnectionError as e:
                print(f"连接错误 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                with self.api_rate_limit_lock:
                    self.consecutive_failures += 1
                    self.failure_count += 1
                
                if attempt < self.max_retries - 1:
                    sleep_time = (2 ** attempt) * random.uniform(2, 4)
                    print(f"等待 {sleep_time:.2f} 秒后重试...")
                    time.sleep(sleep_time)
                else:
                    print(f"最终连接失败: {e}")
                    return ""
                    
            except Exception as e:
                print(f"调用vLLM API时出错 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                with self.api_rate_limit_lock:
                    self.consecutive_failures += 1
                    self.failure_count += 1
                
                if attempt < self.max_retries - 1:
                    sleep_time = (2 ** attempt) * random.uniform(1, 3)
                    print(f"等待 {sleep_time:.2f} 秒后重试...")
                    time.sleep(sleep_time)
                else:
                    print(f"最终失败: {e}")
                    return ""
        
        return ""
    def process_conversation(self, conversation: Dict, idx: int) -> Dict:
        """
        处理单个对话，重新生成助手回答

        Args:
            conversation: 原始对话数据
            idx: 对话索引

        Returns:
            更新后的对话数据
        """
        conversations_list = conversation.get("conversations", [])

        if not conversations_list:
            # 如果使用的是另一种格式
            if "items" in conversation:
                conversations_list = conversation["items"]
            else:
                conversations_list = conversation.get("messages", [])

        # 重新生成的对话
        new_conversations = []
        context = ""

        for i, msg in enumerate(conversations_list):
            role = msg.get("from") or msg.get("role")
            content = msg.get("value") or msg.get("content")

            # 如果是人类消息，保持不变并添加到上下文
            if role in ["human", "user"]:
                new_conversations.append({
                    "from": "human",
                    "value": content
                })
                context += f"Human: {content}\n\n"

            # 如果是助手消息，使用vLLM重新生成
            elif role in ["gpt", "assistant"]:
                # 准备提示，包含之前的上下文
                prompt = f"{context}Assistant:"

                # 调用vLLM生成新回答
                new_response = self.call_vllm_api(prompt)

                new_conversations.append({
                    "from": "gpt",
                    "value": new_response
                })

                # 更新上下文，添加新的回答
                context += f"Assistant: {new_response}\n\n"

        # 保存重新生成的对话
        new_data = {
            "id": conversation.get("id", f"regenerated_{idx}"),
            "conversations": new_conversations
        }

        return new_data

    def worker_task(self, task_queue: queue.Queue, results: List[Dict], 
                   progress_bar: tqdm, processed_count: List[int]):
        """
        工作线程任务函数

        Args:
            task_queue: 任务队列
            results: 结果列表
            progress_bar: 进度条
            processed_count: 已处理的计数
        """
        while True:
            try:
                task = task_queue.get(block=False)
                if task is None:
                    break

                idx, conversation = task
                result = self.process_conversation(conversation, idx)

                # 添加结果到结果列表（线程安全）
                with self.result_lock:
                    results.append(result)

                # 更新进度（线程安全）
                with self.progress_lock:
                    processed_count[0] += 1
                    progress_bar.update(1)

                task_queue.task_done()

            except queue.Empty:
                break
            except Exception as e:
                print(f"处理对话时出错: {e}")
                task_queue.task_done()

    def regenerate_responses_parallel(self, conversations: List[Dict], start_index: int = 0, 
                                    end_index: Optional[int] = None, batch_save: int = 10) -> List[Dict]:
        """
        使用多线程并行重新生成对话中的助手回答

        Args:
            conversations: 原始对话数据列表
            start_index: 起始处理索引
            end_index: 结束处理索引（不含）
            batch_save: 每处理多少条对话保存一次

        Returns:
            更新后的对话数据列表
        """
        if end_index is None:
            end_index = len(conversations)

        total_conversations = end_index - start_index
        print(f"开始并行处理数据，总共 {total_conversations} 条对话，使用 {self.num_threads} 个线程")

        # 初始化结果列表和进度条
        results = []
        progress_bar = tqdm(total=total_conversations)
        processed_count = [0]  # 使用列表以便在函数间传递引用

        # 创建任务队列
        task_queue = queue.Queue()
        for idx in range(start_index, end_index):
            task_queue.put((idx, conversations[idx]))

        # 启动工作线程
        threads = []
        for _ in range(min(self.num_threads, total_conversations)):
            thread = threading.Thread(
                target=self.worker_task,
                args=(task_queue, results, progress_bar, processed_count)
            )
            thread.start()
            threads.append(thread)

        # 定期检查是否需要保存批次
        last_save_count = 0
        while processed_count[0] < total_conversations:
            time.sleep(5)  # 每5秒检查一次

            # 如果处理了足够多的新对话，保存批次
            with self.result_lock:
                current_results = results.copy()

            if len(current_results) >= last_save_count + batch_save:
                self.save_batch(
                    current_results[last_save_count:],
                    start_index + last_save_count,
                    start_index + len(current_results) - 1
                )
                last_save_count = len(current_results)

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        progress_bar.close()
        print(f"并行处理完成，共处理 {len(results)} 条对话")

        # 确保结果按索引排序
        results.sort(key=lambda x: int(x["id"].split("_")[-1]) if "_" in x["id"] else 0)

        # 保存剩余结果
        if last_save_count < len(results):
            self.save_batch(
                results[last_save_count:],
                start_index + last_save_count,
                start_index + len(results) - 1
            )

        return results

    def save_batch(self, data: List[Dict], start_idx: int, end_idx: int):
        """
        保存批量处理的数据

        Args:
            data: 要保存的数据
            start_idx: 批次起始索引
            end_idx: 批次结束索引
        """
        batch_file = os.path.join(self.output_dir, f"regenerated_{start_idx}_to_{end_idx}.json")
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 同时保存为jsonl格式
        batch_jsonl = os.path.join(self.output_dir, f"regenerated_{start_idx}_to_{end_idx}.jsonl")
        with open(batch_jsonl, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"保存批次 {start_idx} 到 {end_idx} 完成，共 {len(data)} 条对话")
    def regenerate_responses_with_threadpool(self, conversations: List[Dict], start_index: int = 0,
                                    end_index: Optional[int] = None, batch_save: int = 10) -> List[Dict]:
            """
            使用线程池重新生成对话中的助手回答

            Args:
                conversations: 原始对话数据列表
                start_index: 起始处理索引
                end_index: 结束处理索引（不含）
                batch_save: 每处理多少条对话保存一次

            Returns:
                更新后的对话数据列表
            """
            if end_index is None:
                end_index = len(conversations)

            total_conversations = end_index - start_index
            print(f"开始使用线程池处理数据，总共 {total_conversations} 条对话，使用 {self.num_threads} 个线程")

            # 准备任务
            tasks = [(idx, conversations[idx]) for idx in range(start_index, end_index)]
            results = []

            # 统计监控
            start_time = time.time()
            last_print_time = start_time
            
            # 使用线程池执行任务
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # 提交所有任务
                future_to_idx = {executor.submit(self.process_conversation, conv, idx): (idx, i) 
                                for i, (idx, conv) in enumerate(tasks)}

                # 处理完成的任务结果
                for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(tasks)):
                    idx, task_idx = future_to_idx[future]
                    try:
                        result = future.result()
                        results.append(result)

                        # 每10秒打印一次统计信息
                        current_time = time.time()
                        if current_time - last_print_time >= 10:
                            with self.api_rate_limit_lock:
                                success_rate = self.success_count / (self.success_count + self.failure_count) * 100 if (self.success_count + self.failure_count) > 0 else 0
                                avg_time_per_item = (current_time - start_time) / len(results) if results else 0
                                print(f"\n统计信息: 成功率={success_rate:.1f}%, 平均处理时间={avg_time_per_item:.2f}s/条, 动态间隔={self.dynamic_interval:.2f}s")
                                print(f"成功={self.success_count}, 失败={self.failure_count}, 连续失败={self.consecutive_failures}")
                            last_print_time = current_time

                        # 每完成batch_save个任务保存一次
                        if len(results) % batch_save == 0:
                            # 获取当前批次的索引范围
                            completed_indices = sorted([future_to_idx[f][0] for f in future_to_idx if f.done()])
                            if completed_indices:
                                batch_start = min(completed_indices)
                                batch_end = max(completed_indices)
                                # 保存当前批次
                                current_batch = [r for r in results if int(r["id"].split("_")[-1]) <= batch_end 
                                                and int(r["id"].split("_")[-1]) >= batch_start]
                                if current_batch:
                                    self.save_batch(current_batch, batch_start, batch_end)

                    except Exception as e:
                        print(f"处理任务 {idx} 时出错: {e}")

            # 打印最终统计信息
            end_time = time.time()
            total_time = end_time - start_time
            with self.api_rate_limit_lock:
                success_rate = self.success_count / (self.success_count + self.failure_count) * 100 if (self.success_count + self.failure_count) > 0 else 0
                avg_time_per_item = total_time / len(results) if results else 0
                print(f"\n最终统计:")
                print(f"总时间: {total_time:.2f}s")
                print(f"成功率: {success_rate:.1f}%")
                print(f"平均处理时间: {avg_time_per_item:.2f}s/条")
                print(f"成功: {self.success_count}, 失败: {self.failure_count}")
                print(f"最终动态间隔: {self.dynamic_interval:.2f}s")

            # 确保结果按索引排序
            results.sort(key=lambda x: int(x["id"].split("_")[-1]) if "_" in x["id"] else 0)

            # 保存剩余结果
            if results:
                remaining_start = start_index + (len(results) // batch_save) * batch_save
                remaining_end = start_index + len(results) - 1
                remaining_batch = results[remaining_start - start_index:]
                if remaining_batch:
                    self.save_batch(remaining_batch, remaining_start, remaining_end)

            return results
    def regenerate_responses(self, conversations: List[Dict], start_index: int = 0, 
                             end_index: Optional[int] = None, batch_save: int = 10) -> List[Dict]:
        """
        单线程重新生成对话中的助手回答（保留此方法作为备选）

        Args:
            conversations: 原始对话数据列表
            start_index: 起始处理索引
            end_index: 结束处理索引（不含）
            batch_save: 每处理多少条对话保存一次

        Returns:
            更新后的对话数据列表
        """
        if end_index is None:
            end_index = len(conversations)

        print(f"开始处理数据，总共 {end_index - start_index} 条对话")
        regenerated_data = []

        for idx in tqdm(range(start_index, end_index)):
            conversation = conversations[idx]
            new_data = self.process_conversation(conversation, idx)
            regenerated_data.append(new_data)

            # 每处理一定量的数据保存一次
            if (idx - start_index + 1) % batch_save == 0:
                self.save_batch(regenerated_data, start_index, idx)

        # 保存剩余数据
        if regenerated_data and (end_index - start_index) % batch_save != 0:
            self.save_batch(regenerated_data, start_index, end_index - 1)

        return regenerated_data
    def merge_batches(self):
        """合并所有的批次文件到一个完整的数据集文件"""
        all_data = []
        batch_files = [f for f in os.listdir(self.output_dir) if f.startswith("regenerated_") and f.endswith(".json")]

        for batch_file in sorted(batch_files, key=lambda x: int(x.split('_')[1])):
            with open(os.path.join(self.output_dir, batch_file), 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                all_data.extend(batch_data)

        # 保存完整数据集
        complete_file = os.path.join(self.output_dir, "regenerated_complete.json")
        with open(complete_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)

        # 保存jsonl格式
        complete_jsonl = os.path.join(self.output_dir, "regenerated_complete.jsonl")
        with open(complete_jsonl, 'w', encoding='utf-8') as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"合并完成，总共 {len(all_data)} 条对话")
def main():
    parser = argparse.ArgumentParser(description='使用vLLM重新生成ShareGPT数据集的回答')
    parser.add_argument('--dataset', type=str, required=True, help='ShareGPT数据集文件路径')
    parser.add_argument('--vllm_url', type=str, required=True, help='vLLM服务的API地址')
    parser.add_argument('--output_dir', type=str, default='./regenerated_data', help='输出目录')
    parser.add_argument('--temperature', type=float, default=0.6, help='生成文本的温度参数')
    parser.add_argument('--max_tokens', type=int, default=2048, help='每次生成的最大token数')
    parser.add_argument('--start_index', type=int, default=0, help='起始处理索引')
    parser.add_argument('--end_index', type=int, default=None, help='结束处理索引（不含）')
    parser.add_argument('--batch_save', type=int, default=10, help='每处理多少条对话保存一次')
    parser.add_argument('--threads', type=int, default=4, help='并行处理的线程数')
    parser.add_argument('--timeout', type=int, default=60, help='API请求超时时间(秒)')
    parser.add_argument('--mode', type=str, default='threadpool', 
                        choices=['single', 'parallel', 'threadpool'],
                        help='处理模式：单线程(single)、自定义多线程(parallel)、线程池(threadpool)')

    args = parser.parse_args()

    regenerator = ShareGPTReGenerator(
        vllm_url=args.vllm_url,
        output_dir=args.output_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_threads=args.threads,
        request_timeout=args.timeout
    )
    if 'ultra' in args.dataset:
        dataset = regenerator.load_ultra_dataset(args.dataset)
        print(f"加载Ultra数据集: {args.dataset}")
    else:
    # 加载数据集
        print(f"加载ShareGPT数据集: {args.dataset}")
        dataset = regenerator.load_sharegpt_dataset(args.dataset)
    print(f"数据集加载完成，共 {len(dataset)} 条对话")

    # 根据所选模式处理数据
    if args.mode == 'single':
        # 单线程处理
        regenerator.regenerate_responses(
            conversations=dataset,
            start_index=args.start_index,
            end_index=args.end_index,  
            batch_save=args.batch_save
        )
    elif args.mode == 'parallel':
        # 自定义多线程处理
        regenerator.regenerate_responses_parallel(
            conversations=dataset,
            start_index=args.start_index,
            end_index=args.end_index,  
            batch_save=args.batch_save
        )
    else:  # threadpool
        # 线程池处理（默认）
        regenerator.regenerate_responses_with_threadpool(
            conversations=dataset,
            start_index=args.start_index,
            end_index=args.end_index,  
            batch_save=args.batch_save
        )

    # 合并批次文件
    regenerator.merge_batches()
if __name__ == "__main__":
    main()