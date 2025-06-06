import re
import matplotlib.pyplot as plt

# 设置你的日志文件路径
log_file_path = "E:/数据收集/无gan/efficient-b0/report/Dataset/efficientnet-b0/seed33/lr0.01_wd5e-4.txt"  # 请替换为实际路径
#E:/数据收集/无gan/efficient-b1/report/Dataset/efficientnet-b1/seed33/lr0.01_wd5e-4.txt
# 初始化数据列表
epochs, train_loss, train_acc, val_loss, val_acc, lrs = [], [], [], [], [], []

# 正则表达式匹配行
pattern = re.compile(
    r"Epoch: (\d+).*?Train Loss: ([\d.]+) Acc: ([\d.]+).*?Val Loss: ([\d.]+) Acc: ([\d.]+).*?LR: ([\d.]+)"
)

# 解析日志文件
with open(log_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            train_loss.append(float(match.group(2)))
            train_acc.append(float(match.group(3)))
            val_loss.append(float(match.group(4)))
            val_acc.append(float(match.group(5)))
            lrs.append(float(match.group(6)))

# 创建竖直排布的子图
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Loss 曲线
axes[0].plot(epochs, train_loss, label='Train Loss')
axes[0].plot(epochs, val_loss, label='Val Loss')
axes[0].set_title("Loss Curve")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

# Accuracy 曲线
axes[1].plot(epochs, train_acc, label='Train Acc')
axes[1].plot(epochs, val_acc, label='Val Acc')
axes[1].set_title("Accuracy Curve")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True)

# Learning Rate 曲线
axes[2].plot(epochs, lrs, label='Learning Rate', color='orange')
axes[2].set_title("Learning Rate Schedule")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("LR")
axes[2].legend()
axes[2].grid(True)

# 自动调整排版并保存
plt.tight_layout()
plt.savefig("training_curves_vertical.png", dpi=300)
print("✅ 竖排图已保存为 training_curves_vertical.png")
