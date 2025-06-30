# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# # plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示字体
# # plt.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题
# def logarithmic_step_scheduler(epoch, max_epochs):
#     """对数阶梯调度：教师对数降温，学生阶梯调整"""
#     T_t = np.exp(-1.5 * epoch / max_epochs) * 2.5 + 0.3
#     T_t = max(T_t, 1.0)
#     if epoch < max_epochs * 0.25:    # 第一阶段：基础学习
#         T_s = 1.3
#     elif epoch < max_epochs * 0.6:   # 第二阶段：增强探索
#         T_s = 2.0
#     else:                            # 第三阶段：精炼提升
#         T_s = 0.6
#     return T_t, T_s

# # 模拟测试
# max_epochs = 10
# epochs = np.arange(0, max_epochs + 1)
# T_t_vals = []
# T_s_vals = []

# for epoch in epochs:
#     T_t, T_s = logarithmic_step_scheduler(epoch, max_epochs)
#     T_t_vals.append(T_t)
#     T_s_vals.append(T_s)

# # 绘制并保存图像
# plt.figure()
# plt.plot(epochs, T_t_vals, label='Student_Temp $T_t$')
# plt.plot(epochs, T_s_vals, label='Teacher_Temp $T_s$')
# plt.xlabel('Epoch')
# plt.ylabel('Temperature')
# plt.legend()
# plt.title('Temp Curve')
# plt.grid(True)
# plt.savefig('/root/autodl-tmp/save/temp')
# plt.show()



import numpy as np
import matplotlib.pyplot as plt

def logarithmic_step_scheduler(epoch, max_epochs):
    """对数阶梯调度：教师对数降温，学生阶梯调整"""
    T_t = np.exp(-1.5 * epoch / max_epochs) * 2.5 + 0.3
    T_t = max(T_t, 1.0)
    if epoch < max_epochs * 0.25:    # 第一阶段：基础学习
        T_s = 1.3
    elif epoch < max_epochs * 0.6:   # 第二阶段：增强探索
        T_s = 2.0
    else:                            # 第三阶段：精炼提升
        T_s = 0.6
    return T_t, T_s

# 模拟测试
max_epochs = 10
epochs = np.arange(0, max_epochs + 1)
T_t_vals = []
T_s_vals = []

for epoch in epochs:
    T_t, T_s = logarithmic_step_scheduler(epoch, max_epochs)
    T_t_vals.append(T_t)
    T_s_vals.append(T_s)

# --- 单独绘制并保存教师温度图像 ---
plt.figure()
plt.plot(epochs, T_t_vals, label='Teacher_Temp $T_t$', color='blue')
plt.xlabel('Epoch',fontsize=18)
plt.ylabel('Temperature',fontsize=18)
plt.title('Teacher Temperature Curve',fontsize=18)
plt.grid(True)
plt.legend()
plt.savefig('/root/autodl-tmp/save/results/gpt2/image/teacher_temp.png', dpi=300, bbox_inches='tight')
plt.close()

# --- 单独绘制并保存学生温度图像 ---
plt.figure()
plt.plot(epochs, T_s_vals, label='Student_Temp $T_s$', color='green')
plt.xlabel('Epoch',fontsize=18)
plt.ylabel('Temperature',fontsize=18)
plt.title('Student Temperature Curve',fontsize=18)
plt.grid(True)
plt.legend()
plt.savefig('/root/autodl-tmp/save/results/gpt2/image/student_temp.png', dpi=300, bbox_inches='tight')
plt.close()