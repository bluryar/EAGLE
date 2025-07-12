#!/bin/bash

# 优化后的数据生成脚本
# 提供多种并发策略以避免API过载

echo "=== EAGLE数据生成优化版 ==="
echo "选择并发策略:"
echo "1. 保守策略 (8线程, 适合低配置或网络不稳定环境)"
echo "2. 平衡策略 (16线程, 推荐使用)"
echo "3. 激进策略 (32线程, 适合高配置环境)"
echo "4. 自定义策略"

read -p "请选择策略 (1-4): " strategy

case $strategy in
    1)
        threads=8
        timeout=180
        echo "使用保守策略: 8线程, 180秒超时"
        ;;
    2)
        threads=16
        timeout=120
        echo "使用平衡策略: 16线程, 120秒超时"
        ;;
    3)
        threads=32
        timeout=90
        echo "使用激进策略: 32线程, 90秒超时"
        ;;
    4)
        read -p "请输入线程数: " threads
        read -p "请输入超时时间(秒): " timeout
        echo "使用自定义策略: ${threads}线程, ${timeout}秒超时"
        ;;
    *)
        echo "无效选择，使用默认平衡策略"
        threads=16
        timeout=120
        ;;
esac

echo "开始执行数据生成..."
python /root/EAGLE/eagle/data/gen.py  \
  --dataset /root/EAGLE/eagle/data/raw/openo1.jsonl   \
  --vllm_url http://127.0.0.1:8000/v1/chat/completions \
  --output_dir /root/EAGLE/eagle/data/save \
  --threads $threads \
  --timeout $timeout \
  --batch_save 180000 \
  --mode threadpool

echo "数据生成完成!" 