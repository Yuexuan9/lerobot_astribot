# lerobot_astribot

一个用于将 Astribot ROS Bag 数据转换为 LeRobot 3.0 格式的工具，支持训练和推理。

## 数据转换流程

```
ROS Bag (250Hz 关节数据 + 30Hz 图像数据)
         ↓
convert.py / convert_tar.py
         ↓
LeRobot 3.0 格式 (30 FPS，同步帧)
         ↓
训练 / 推理
```

## 输出数据格式

转换后生成标准 LeRobot 数据集，包含：

- **状态向量**: 25 维 (7+7+1+1+2+4+3)
- **动作向量**: 25 维
- **图像**: `head`, `torso`, `wrist_left`, `wrist_right`
- **FPS**: 30

## 安装要求

```bash
# 创建并激活 conda 环境
conda create -n lerobot python=3.10
conda activate lerobot

# 安装依赖（根据实际需求添加）
pip install lerobot
```

## 使用方法

### 1. 数据转换

#### 从 ROS Bag 转换

将所有 episodes 合并到单个数据集：

```bash
python convert.py /xxx -o ./output/xxx --repo-id astribot/[repo-id]
```


### 2. 数据回放测试

在训练前，可以先测试数据是否正确：

```bash
python3 examples/A01_test.py \
    /root/ros_ws/src/astribot_sdk-main/examples/raw_data.bag \
    --num-replays 2
```

### 3. 模型训练

使用多 GPU 训练 ACT 策略：

```bash
conda activate lerobot

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --multi_gpu \
    --num_processes=4 \
    --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=astribot/test002 \
    --dataset.root=/workspace/astribot_lerobot_v30_test002 \
    --policy.type=act \
    --policy.push_to_hub=false \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.vision_backbone=resnet18 \
    --steps=1000000 \
    --batch_size=10 \
    --save_freq=5000 \
    --num_workers=16 \
    --output_dir=/root/outputs_4gpu_1231_lr4e5max \
    --use_policy_training_preset=false \
    --optimizer.type=adamw \
    --optimizer.lr=4e-5 \
    --optimizer.weight_decay=0.0001 \
    --scheduler.type=diffuser \
    --scheduler.name=cosine \
    --scheduler.num_warmup_steps=1000 \
    --wandb.enable=true \
    --wandb.project=astribot-act-training-1231_lr4e5max
```

#### 训练参数说明

| 参数 | 说明 |
|------|------|
| `--num_processes` | GPU 数量 |
| `--mixed_precision` | 混合精度训练（bf16） |
| `--policy.chunk_size` | 动作序列长度 |
| `--policy.n_action_steps` | 动作执行步数 |
| `--policy.vision_backbone` | 视觉编码器（resnet18） |
| `--batch_size` | 每个 GPU 的批次大小 |
| `--optimizer.lr` | 学习率 |
| `--optimizer.weight_decay` | 权重衰减 |
| `--save_freq` | 模型保存频率（步数） |
| `--num_workers` | 数据加载线程数 |
| `--wandb.enable` | 启用 Weights & Biases 日志 |


## 数据格式详解

### 状态向量 (25 维)

- 左臂关节位置: 7 维
- 右臂关节位置: 7 维
- 左夹爪状态: 1 维
- 右夹爪状态: 1 维
- 躯干关节: 2 维
- 底盘姿态: 4 维
- 头部姿态: 3 维

### 动作向量 (25 维)

动作向量与状态向量维度对应，表示机器人下一时刻的目标状态。

### 图像数据

- `head`: 头部摄像头
- `torso`: 躯干摄像头
- `wrist_left`: 左腕摄像头
- `wrist_right`: 右腕摄像头

所有图像统一采样至 30 FPS。

## 注意事项

1. **数据同步**: 转换时会将 250Hz 的关节数据下采样到 30Hz，与图像数据同步
2. **存储空间**: 确保有足够的磁盘空间存储转换后的数据集
3. **GPU 内存**: 训练时根据 GPU 内存调整 `batch_size` 和 `num_processes`
4. **WandB 配置**: 首次使用需要登录 `wandb login`

## 平台集成

该工具也支持平台自动转换为 LeRobot 3.0 格式，无需手动执行转换脚本。

## 问题排查

### 转换问题

- **转换失败**: 检查 ROS Bag 文件完整性和路径是否正确
- **图像同步错误**: 确认 bag 文件包含所有必需的图像话题
- **维度不匹配**: 验证状态向量和动作向量的维度是否为 25

### 训练问题

- **训练 OOM**: 减小 `batch_size` 或使用更少的 GPU
- **数据加载慢**: 增加 `num_workers` 数量（注意 CPU 核心数）
- **loss 不收敛**: 调整学习率或检查数据质量

### 性能优化

- 使用 SSD 存储数据集以提高 I/O 速度
- 启用混合精度训练减少内存占用
- 根据 GPU 型号调整 batch_size 以最大化利用率



## 联系方式

如有问题，请联系项目维护者。
