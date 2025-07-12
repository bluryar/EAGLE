import argparse
import deepspeed
import shutil

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/root/Qwen3-4B')
parser.add_argument('--trainpath', type=str,
                    default="/root/EAGLE/eagle/data/save/regenerated_data/train.jsonl")
parser.add_argument('--testpath', type=str,
                    default="/root/EAGLE/eagle/data/save/regenerated_data/test.jsonl")
parser.add_argument('--savedir', type=str, default='0')
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()
import json
import re

deepspeed_config = args.deepspeed_config
with open(deepspeed_config) as f:
    ds_config = json.load(f)
train_config = {
    "bs": ds_config["train_micro_batch_size_per_gpu"],
    "num_epochs": 1,
    "num_workers": 2,
    "max_len": 2048,
    "config_path": "config.json",
}

from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from cnets import padding

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate.utils import set_seed

set_seed(0)
from cnets import Model
from configs import EConfig
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
# import accelerate
import numpy as np
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup



def build_dataset_rank(
        tokenizer, datapath
):
    import torch.distributed as dist
    
    # 添加数据集缓存机制
    cache_path = f"dataset_cache_{datapath.split('/')[-1]}.pt"
    
    # 获取当前进程的 rank
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    # 只有 rank 0 负责数据预处理
    if rank == 0:
        if os.path.exists(cache_path):
            try:
                print(f"Loading cached dataset from {cache_path}")
                cached_dataset = torch.load(cache_path, weights_only=False)
                # 检查缓存是否有效
                if hasattr(cached_dataset, '__len__') and len(cached_dataset) > 0:
                    pass  # 缓存有效
                else:
                    raise ValueError("Invalid cached dataset")
            except Exception as e:
                print(f"Failed to load cached dataset: {e}")
                print("Rebuilding dataset...")
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                cached_dataset = None
        else:
            cached_dataset = None
            
        if cached_dataset is None:
            print(f"Processing dataset {datapath}...")
            ds = load_dataset('json', data_files=datapath)
            ds = ds['train']
            ds = ds.shuffle(seed=42)
            ds1 = ds
            original_columns1 = ds1.column_names
            num_proc = 28

            def preprocess_function(examples):
                new_examples = {
                    "attention_mask": [],
                    "input_ids": [],
                    "loss_mask": []
                }
                for i in range(len(examples['id'])):
                    messages = [
                        {"role": "system",
                         "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
                    ]
                    convroles = ["user", "assistant"]
                    roles = {"human": "user", "gpt": "assistant"}
                    source = examples['conversations'][i]
                    if not source:
                        continue
                    if roles[source[0]["from"]] != "user":
                        # Skip the first one if it is not from human
                        source = source[1:]
                    for j, sentence in enumerate(source):
                        role = roles[sentence["from"]]
                        assert role == convroles[j % 2], f"{i}"
                        # if sentence["from"]=="gpt":
                        #     sentence["value"]=" "+sentence["value"]
                        messages.append(
                            {"role": role, "content": sentence["value"]}
                        )
                    conversation = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )

                    if not tokenizer.pad_token_id:
                        tokenizer.pad_token_id = tokenizer.unk_token_id

                    input_ids = tokenizer(
                        conversation,
                        return_tensors="pt",
                        max_length=2048,
                        truncation=True,
                        add_special_tokens=False,
                    ).input_ids[0]
                    loss_mask = torch.ones_like(input_ids)
                    # print(i)
                    # TODO

                    sep = "<|im_start|>assistant"

                    total_len = len(input_ids)

                    sep2 = "<|im_start|>user"
                    turns = conversation.split(sep2)
                    # if len(turns) == 1:
                    #     continue
                    turns[1] = turns[0] + sep2 + turns[1]
                    turns = turns[1:]

                    cur_len = 0
                    # loss_mask[:cur_len] = 0
                    # TODO
                    for i, turn in enumerate(turns):
                        if turn == "":
                            break
                        turn_len = len(tokenizer(turn).input_ids)

                        parts = turn.split(sep)
                        if len(parts) != 2:
                            break
                        parts[0] += sep
                        # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                        instruction_len = len(tokenizer(parts[0]).input_ids) - 1

                        # Ignore the user instructions
                        if i == 0:
                            loss_mask[cur_len: cur_len + instruction_len - 2] = 0
                        else:
                            loss_mask[cur_len - 3: cur_len + instruction_len + 1] = 0
                        cur_len += turn_len
                        if i != 0:
                            cur_len += 3
                        # cur_len+=2

                        # if i != 0 and not tokenizer.legacy:
                        #     # The legacy and non-legacy modes handle special tokens differently
                        #     cur_len -= 1

                    loss_mask[cur_len:] = 0
                    attention_mask = torch.ones_like(loss_mask)

                    # new_examples["conversation"].append(conversation)
                    new_examples["input_ids"].append(input_ids[None, :])
                    new_examples["loss_mask"].append(loss_mask[None, :])
                    new_examples["attention_mask"].append(attention_mask[None, :])

                return new_examples

            ds1 = ds1.map(
                preprocess_function,
                batched=True,
                num_proc=num_proc,
                remove_columns=original_columns1,
                load_from_cache_file=False
            )

            ds1.set_format(type="torch")
            
            # 缓存处理后的数据集
            try:
                torch.save(ds1, cache_path)
                print(f"Dataset cached to {cache_path}")
                cached_dataset = ds1
            except Exception as e:
                print(f"Failed to cache dataset: {e}")
                cached_dataset = ds1
        
        # 将缓存的数据集设为返回值
        final_dataset = cached_dataset
    
    # 同步所有进程，确保 rank 0 完成缓存
    if dist.is_initialized():
        dist.barrier()
    
    # 现在所有进程都可以加载缓存
    if rank != 0:
        # 等待缓存文件创建（防止竞争条件）
        if not os.path.exists(cache_path):
            import time
            while not os.path.exists(cache_path):
                time.sleep(1)
        
        print(f"Rank {rank}: Loading cached dataset from {cache_path}")
        try:
            final_dataset = torch.load(cache_path, weights_only=False)
        except Exception as e:
            print(f"Rank {rank}: Failed to load cached dataset: {e}")
            # 如果加载失败，重新处理数据集（不缓存）
            ds = load_dataset('json', data_files=datapath)
            ds = ds['train']
            ds = ds.shuffle(seed=42)
            final_dataset = ds
    
    return final_dataset


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['input_ids'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_attention_mask = torch.cat(
            [self.paddingtensor2D(item['attention_mask'], max_length) for item in features])
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item['loss_mask'], max_length) for item in features])

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


tokenizer = AutoTokenizer.from_pretrained(args.basepath)
traindataset = build_dataset_rank(tokenizer, args.trainpath)
testdataset = build_dataset_rank(tokenizer, args.testpath)

config = EConfig.from_pretrained(train_config["config_path"])
model = Model(config, path=args.basepath, load_emb=True, load_head=True)
model.scandata(args.trainpath, args.basepath)


criterion = nn.SmoothL1Loss(reduction="none")

num_epochs = train_config["num_epochs"]

model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                     model=model,
                                                     model_parameters=model.parameters(),
                                                     )

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()
if global_rank == 0:
    import swanlab

    # SwanLab 初始化，使用项目名称和配置
    swanlab.init(project="eagle3-qwen-4b", config=ds_config)

os.makedirs(args.savedir, exist_ok=True)

sampler = DistributedSampler(testdataset, num_replicas=world_size, rank=global_rank, shuffle=False)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], sampler=sampler, num_workers=4, pin_memory=True,
                         collate_fn=DataCollatorWithPadding())

train_sampler = DistributedSampler(traindataset, num_replicas=world_size, rank=global_rank, shuffle=True)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], sampler=train_sampler, num_workers=4,
                          pin_memory=True,
                          collate_fn=DataCollatorWithPadding())


def find_max_state_with_file(directory, filename="zero_to_fp32.py"):
    max_a = -1
    valid_checkpoint_found = False
    
    for subdir in os.listdir(directory):
        match = re.match(r"state_(\d+)", subdir)
        if match:
            a_value = int(match.group(1))
            subdir_path = os.path.join(directory, subdir)
            file_path = os.path.join(subdir_path, filename)
            
            # 检查是否为有效的 DeepSpeed 检查点目录
            if os.path.isdir(subdir_path) and os.path.exists(file_path):
                # 进一步检查是否存在其他必要的检查点文件
                required_files = ["zero_to_fp32.py", "latest"]
                mp_rank_files = [f for f in os.listdir(subdir_path) if f.startswith("mp_rank_")]
                
                # 检查是否有基本的检查点结构
                if len(mp_rank_files) > 0 or any(os.path.exists(os.path.join(subdir_path, rf)) for rf in required_files):
                    if a_value > max_a:
                        max_a = a_value
                        valid_checkpoint_found = True
    
    if not valid_checkpoint_found:
        return None, 0
    return f"{directory}/state_{max_a}", max_a + 1


checkpoint_path, start_epoch = find_max_state_with_file(args.savedir)
if checkpoint_path:
    print(f"load from {checkpoint_path}")
    try:
        model_engine.load_checkpoint(checkpoint_path)
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        print(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        print("Starting training from scratch...")
        start_epoch = 0
else:
    print("No valid checkpoint found, starting training from scratch...")
    start_epoch = 0

# 初始化最佳验证准确率
best_val_acc = -1.0
best_epoch = -1

for epoch in range(start_epoch, num_epochs):
    train_sampler.set_epoch(epoch+1)
    print(f"Now training epoch {epoch}")

    model.train()
    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]


    for batch_idx, data in enumerate(tqdm(train_loader)):

        model.zero_grad()

        plosses, vlosses, acces = model_engine(input_ids=data["input_ids"].to(rank),
                                               attention_mask=data["attention_mask"].to(rank),
                                               loss_mask=data["loss_mask"],
                                               )

        ploss_weight = [0.8 ** i for i in range(len(plosses))]
        ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
        loss = ploss
        model_engine.backward(loss)


        model_engine.step()

        if global_rank == 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"]}
            for i in range(len(plosses)):
                logdict[f"train/ploss_{i}"] = plosses[i].item()
            for i in range(len(acces)):
                logdict[f"train/acc_{i}"] = acces[i]
            swanlab.log(logdict)
        epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
        epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]


    for i in range(len(epoch_acces)):
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()
        if global_rank == 0:
            swanlab.log({f"train/epochacc_{i}": acc_i})
            print(f"Train Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        if global_rank == 0:
            swanlab.log({f"train/epochploss_{i}": loss_i})
            print(f"Train Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}")

    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]

    for batch_idx, data in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            plosses, vlosses, acces = model_engine(input_ids=data["input_ids"].to(rank),
                                                   attention_mask=data["attention_mask"].to(rank),
                                                   loss_mask=data["loss_mask"],
                                                   )
            epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
            epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]

    for i in range(len(epoch_acces)):
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()
        if global_rank == 0:
            swanlab.log({f"test/epochacc_{i}": acc_i})
            print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        if global_rank == 0:
            swanlab.log({f"test/epochploss_{i}": loss_i})
            print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}")

    # 计算当前epoch的平均验证准确率（用于判断是否为best）
    current_val_acc = 0.0
    for i in range(len(epoch_acces)):
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        current_val_acc += acc_i.item()
    current_val_acc /= len(epoch_acces)  # 平均准确率

    # 保存最新检查点
    model_engine.save_16bit_model(f"{args.savedir}/state_latest", exclude_frozen_parameters=True)
    if epoch % 10 == 0:
        deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f"{args.savedir}/state_latest")

    # 检查是否为最佳模型
    is_best = current_val_acc > best_val_acc
    if is_best:
        best_val_acc = current_val_acc
        best_epoch = epoch
        if global_rank == 0:
            print(f"New best model found at epoch {epoch + 1} with validation accuracy: {best_val_acc:.4f}")
        
        # 保存最佳检查点
        model_engine.save_16bit_model(f"{args.savedir}/state_best", exclude_frozen_parameters=True)
        if epoch % 10 == 0:
            deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f"{args.savedir}/state_best")

    # 清理旧的检查点，只保留 latest 和 best
    if global_rank == 0:
        for item_name in os.listdir(args.savedir):
            item_path = os.path.join(args.savedir, item_name)
            if os.path.isdir(item_path) and item_name.startswith("state_") and item_name not in ["state_latest", "state_best"]:
                print(f"Removing old checkpoint: {item_path}")
                shutil.rmtree(item_path)

# 训练结束后，将 latest 重命名为 final
if global_rank == 0:
    latest_path = f"{args.savedir}/state_latest"
    final_path = f"{args.savedir}/state_final"
    
    if os.path.exists(latest_path):
        if os.path.exists(final_path):
            shutil.rmtree(final_path)
        shutil.move(latest_path, final_path)
        print(f"Renamed latest checkpoint to final: {final_path}")
    
    print(f"Training completed. Best model was at epoch {best_epoch + 1} with validation accuracy: {best_val_acc:.4f}")
    print(f"Final checkpoints saved: state_best and state_final")
