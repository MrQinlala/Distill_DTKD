import matplotlib.pyplot as plt
import numpy as np

# 模拟的概率分布数据
probs = [0.11, 0.14, 0.41, 0.2, 0.15]
colors = ['#e69dac', '#dce6b5', '#e6e1ef', '#b6ddea', '#f2cfb2']

# 创建图形
fig, ax = plt.subplots(figsize=(3, 2))

# 绘制柱状图
bars = ax.bar(range(len(probs)), probs, color=colors, edgecolor='black')

# 去除坐标轴
ax.axis('off')

# 去掉边框线
for spine in ax.spines.values():
    spine.set_visible(False)

# 设置统一底线
ax.axhline(0, color='black', linewidth=1)

plt.tight_layout()
plt.savefig("/root/autodl-tmp/save/results/gpt2/image/stu_bar", dpi=300, bbox_inches='tight')
plt.close()
print("save image")
