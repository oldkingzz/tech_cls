#!/bin/bash

: '
SparseMoE模型训练脚本
功能：
- 训练单流稀疏MoE模型
- 支持多工况轴承数据
- 自动记录实验结果
- 保存最佳模型检查点

使用方法：
1. 直接运行：bash train_SparseMoE.sh
2. 自定义实验名：EXP_NAME="实验名称" bash train_SparseMoE.sh
'

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 如果没有指定实验名称，使用默认值
if [ -z "$EXP_NAME" ]; then
    EXP_NAME="SingleStreamMoE_RUL_AllData"
fi

# 如果没有指定种子，使用默认值
if [ -z "$SEED" ]; then
    SEED=42
fi

# 创建带时间戳和种子的日志目录
LOG_DIR="/root/tf-logs/${EXP_NAME}_seed${SEED}_${TIMESTAMP}"

# 创建必要的目录
mkdir -p /root/autodl-tmp/checkpoints
mkdir -p "$LOG_DIR"

# 清理之前的日志
rm -rf /root/autodl-tmp/tf-logs/*

# 数据集配置
DATA_DIR="/root/autodl-tmp/processed_data"

# 定义训练文件（按工况分组）

TRAIN_FILES="'33Hz27kN_Bearing1_3.pt' '33Hz27kN_Bearing1_4.pt' '33Hz27kN_Bearing2_1.pt' '33Hz27kN_Bearing3_3.pt' \
'37.5Hz11kN_Bearing2_1.pt' '37.5Hz11kN_Bearing2_2.pt' '37.5Hz11kN_Bearing2_3.pt' '37.5Hz11kN_Bearing2_4.pt' '37.5Hz11kN_Bearing2_5.pt' \
'27.5Hz4.2kN_Bearing2_1.pt' '27.5Hz4.2kN_Bearing2_2.pt' '27.5Hz4.2kN_Bearing2_3.pt' '27.5Hz4.2kN_Bearing2_4.pt' '27.5Hz4.2kN_Bearing2_5.pt' '27.5Hz4.2kN_Bearing2_6.pt' '27.5Hz4.2kN_Bearing2_7.pt' \
'35Hz12kN_Bearing1_1.pt' '35Hz12kN_Bearing1_2.pt' '35Hz12kN_Bearing1_3.pt' '35Hz12kN_Bearing1_4.pt' \
'25Hz5kN_Bearing1_1.pt' '25Hz5kN_Bearing1_2.pt' '25Hz5kN_Bearing1_3.pt' '25Hz5kN_Bearing1_4.pt' '25Hz5kN_Bearing1_6.pt' '25Hz5kN_Bearing1_7.pt' \
'40Hz10kN_Bearing3_1.pt' '40Hz10kN_Bearing3_2.pt' '40Hz10kN_Bearing3_3.pt' '40Hz10kN_Bearing3_4.pt' '40Hz10kN_Bearing3_5.pt' \
'30Hz4kN_Bearing3_1.pt' '30Hz4kN_Bearing3_2.pt' '30Hz4kN_Bearing3_3.pt'"
: '
TRAIN_FILES="'33Hz27kN_Bearing1_3.pt' '33Hz27kN_Bearing1_4.pt' '33Hz27kN_Bearing2_1.pt' '33Hz27kN_Bearing3_3.pt' \
'27.5Hz4.2kN_Bearing2_1.pt' '27.5Hz4.2kN_Bearing2_2.pt' '27.5Hz4.2kN_Bearing2_3.pt' '27.5Hz4.2kN_Bearing2_4.pt' '27.5Hz4.2kN_Bearing2_5.pt' '27.5Hz4.2kN_Bearing2_6.pt' '27.5Hz4.2kN_Bearing2_7.pt' \
'25Hz5kN_Bearing1_1.pt' '25Hz5kN_Bearing1_2.pt' '25Hz5kN_Bearing1_3.pt' '25Hz5kN_Bearing1_4.pt' '25Hz5kN_Bearing1_6.pt' '25Hz5kN_Bearing1_7.pt' \
'30Hz4kN_Bearing3_1.pt' '30Hz4kN_Bearing3_2.pt' '30Hz4kN_Bearing3_3.pt'"
'
: '
TRAIN_FILES="
'37.5Hz11kN_Bearing2_1.pt' '37.5Hz11kN_Bearing2_2.pt' '37.5Hz11kN_Bearing2_3.pt' '37.5Hz11kN_Bearing2_4.pt' '37.5Hz11kN_Bearing2_5.pt' \
'35Hz12kN_Bearing1_1.pt' '35Hz12kN_Bearing1_2.pt' '35Hz12kN_Bearing1_3.pt' '35Hz12kN_Bearing1_4.pt' \
'40Hz10kN_Bearing3_1.pt' '40Hz10kN_Bearing3_2.pt' '40Hz10kN_Bearing3_4.pt' '40Hz10kN_Bearing3_5.pt' \
"
'
# 验证和测试文件
VAL_FILES="'40Hz10kN_Bearing3_3.pt'"
TEST_FILES="'35Hz12kN_Bearing1_5.pt'"

# 打印数据集信息
echo "已选择以下训练文件："
echo $TRAIN_FILES | tr " " "\n" | sed "s/'//g"
echo "总文件数：$(echo $TRAIN_FILES | wc -w)"
echo ""

# 模型参数
D_MODEL=128
NUM_HEADS=4
NUM_LAYERS=3
NUM_EXPERTS=4  # 减少专家数量

# 训练参数
BATCH_SIZE=64  # 增大批次大小，因为数据量变大了
EPOCHS=10   # 减少epoch数，因为数据量变大了
LR=5e-4       # 降低学习率
WEIGHT_DECAY=1e-4
WARMUP_STEPS=1000  # 增加预热步数，因为总步数变多了

# 损失权重
RUL_WEIGHT=1.0
MOE_WEIGHT=0.005  # 降低MoE损失权重

echo "========== 实验配置 =========="
echo "实验名称: $EXP_NAME"
echo ""
echo "========== 数据集配置 =========="
echo "数据目录: $DATA_DIR"
echo "训练文件: $TRAIN_FILES"
echo "验证文件: $VAL_FILES"
echo "测试文件: $TEST_FILES"
echo ""
echo "========== 模型配置 =========="
echo "模型维度: $D_MODEL"
echo "注意力头数: $NUM_HEADS"
echo "层数: $NUM_LAYERS"
echo "专家数量: $NUM_EXPERTS"
echo ""
echo "========== 训练配置 =========="
echo "批次大小: $BATCH_SIZE"
echo "训练轮数: $EPOCHS"
echo "学习率: $LR"
echo "权重衰减: $WEIGHT_DECAY"
echo "预热步数: $WARMUP_STEPS"
echo "RUL损失权重: $RUL_WEIGHT"
echo "MoE损失权重: $MOE_WEIGHT"
echo ""

# 保存目录
SAVE_DIR="/root/autodl-tmp/checkpoints/${EXP_NAME}_${TIMESTAMP}"
mkdir -p "$SAVE_DIR"

# 运行训练脚本
python /root/YOTO/main_SparseMoE.py \
    --d_model $D_MODEL \
    --num_heads $NUM_HEADS \
    --num_layers $NUM_LAYERS \
    --num_experts $NUM_EXPERTS \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --rul_weight $RUL_WEIGHT \
    --moe_weight $MOE_WEIGHT \
    --data_dir "$DATA_DIR" \
    --train_files $TRAIN_FILES \
    --val_files $VAL_FILES \
    --test_files $TEST_FILES \
    --use_filenames \
    --save_dir "$SAVE_DIR" \
    --exp_name "$EXP_NAME" \
    --log_dir "$LOG_DIR" \
    --seed $SEED

# 打印提示信息
echo "单流MoE训练已启动！"
echo "实验名称: $EXP_NAME"
echo ""
echo "模型配置:"
echo "- 特征维度: 1 (振动特征)"
echo "- 模型维度: $D_MODEL"
echo "- 注意力头数: $NUM_HEADS"
echo "- Transformer层数: $NUM_LAYERS"
echo "- 专家数量: $NUM_EXPERTS"
echo ""
echo "数据集信息:"
echo "训练集: $TRAIN_FILES"
echo "验证集: $VAL_FILES"
echo "测试集: $TEST_FILES"
echo ""
echo "训练参数:"
echo "- 批次大小: $BATCH_SIZE"
echo "- 训练轮数: $EPOCHS"
echo "- 学习率: $LR"
echo "- 预热步数: $WARMUP_STEPS"
echo "- RUL损失权重: $RUL_WEIGHT"
echo "- MoE损失权重: $MOE_WEIGHT"
echo ""
echo "指标记录:"
echo "- TensorBoard日志保存在: $LOG_DIR"
echo "- 检查点保存在: $SAVE_DIR"
echo "  - best_rul.pth: 最佳RUL预测模型" 