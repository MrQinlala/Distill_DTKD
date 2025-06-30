import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def draw_block(ax, x, y, width, height, depth, label=""):
    """画一个模拟3D块的层"""
    # 正面
    front = Rectangle((x, y), width, height, edgecolor='black', facecolor='lightgray')
    ax.add_patch(front)
    # 侧面
    ax.plot([x+width, x+width+depth], [y, y-depth], color='black')
    ax.plot([x+width, x+width+depth], [y+height, y+height-depth], color='black')
    ax.plot([x+width+depth, x+width+depth], [y-depth, y+height-depth], color='black')
    ax.plot([x, x+depth], [y, y-depth], color='black')
    ax.plot([x, x+depth], [y+height, y+height-depth], color='black')
    ax.plot([x+depth, x+depth], [y-depth, y+height-depth], color='black')

    if label:
        ax.text(x + width / 2, y + height / 2, label, fontsize=8, ha='center', va='center')

def draw_model(ax, start_x, start_y, layers, label):
    x = start_x
    for idx, (w, h, d) in enumerate(layers):
        draw_block(ax, x, start_y, w, h, d)
        x += w + 0.5
    ax.text(start_x + 5, start_y + 3, label, fontsize=12, ha='center')

# 图初始化
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 25)
ax.set_ylim(-10, 10)
ax.axis('off')

# 模型结构（每个层的宽、高、深度）
teacher_layers = [(1, 3, 2), (1.2, 3, 2), (1.5, 3, 2), (2, 3, 2), (2, 2.5, 1.5), (1, 2, 1)]
student_layers = [(1, 2.5, 1.5), (1.2, 2.5, 1.5), (1.5, 2.5, 1.5), (1.7, 2.5, 1), (1, 2, 1)]

# 画 Teacher 模型
draw_model(ax, start_x=1, start_y=2, layers=teacher_layers, label="Teacher")

# 画 Student 模型
draw_model(ax, start_x=1, start_y=-5, layers=student_layers, label="Student")
plt.savefig("/root/autodl-tmp/save/results/gpt2/image/structure", dpi=300, bbox_inches='tight')
plt.close()
plt.show()
