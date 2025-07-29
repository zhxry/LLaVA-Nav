import matplotlib.pyplot as plt
import json

log_file = './log/sft_pure.log'
epochs = []
losses = []
accuracies = []

with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.replace("'", '"')
        data = json.loads(line)
        epochs.append(data['epoch'])
        losses.append(data['loss'])
        accuracies.append(data['mean_token_accuracy'])

# 绘制 loss 和 accuracy 曲线
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss', color=color)
ax1.plot(epochs, losses, color=color, marker='o', label='loss')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # 共享 x 轴
color = 'tab:blue'
ax2.set_ylabel('mean_token_accuracy', color=color)
ax2.plot(epochs, accuracies, color=color, marker='x', label='mean_token_accuracy')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Loss & Mean Token Accuracy vs Epoch')
fig.tight_layout()
plt.show()
