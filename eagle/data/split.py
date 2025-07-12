import json
import random
import os

# 读取完整数据集
with open('/root/EAGLE/eagle/data/save/regenerated_complete.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# 随机打乱并分割
random.shuffle(data)
split_idx = int(len(data) * 0.997)  # 99.7% 训练，0.3% 测试

train_data = data[:split_idx]
test_data = data[split_idx:]

# 确保保存目录存在
save_dir = '/root/EAGLE/eagle/data/save/regenerated_data'
os.makedirs(save_dir, exist_ok=True)

# 保存分割后的数据
with open(os.path.join(save_dir, 'train.jsonl'), 'w') as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

with open(os.path.join(save_dir, 'test.jsonl'), 'w') as f:
    for item in test_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')