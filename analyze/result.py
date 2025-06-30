import pandas as pd
import matplotlib.pyplot as plt
import io



"""
base:
teacher	29.3529	14.704	18.481	26.4669
student	21.4654	9.7727	14.423	18.1111
FKL	21.9954	9.9146	15.2773	19.4956
RKL	23.0618	10.0044	15.1316	20.51
JSD	22.1981	10.4161	14.7571	16.5649
SFKL	22.0892	10.045	14.10908	16.00644
SRKL	21.9980	9.69484	14.3313	15.00836
AKL	22.76834	10.0092	14.85192	20.1976
TDKD(ours)	23.2342	10.01118	15.73004	20.7291"""
# 您的更新后的数据字符串
data = """
teacher	29.3529	14.704	18.481	26.4669
student	24.5344	12.5131	16.5319	22.9814
FKL	23.16258	12.8387	16.7185	21.821
RKL	25.0735	13.3808	16.3734	22.8593
JSD	24.6295	13.4318	15.0514	22.0526
SFKL	24.4268	12.97648	15.9143	22.03244
SRKL	24.5557	12.36857	16.51257	22.64253
AKL	25.21092	12.17624	16.18218	24.00718
TDKD(ours)	26.0757	13.8309	16.710	24.1632
"""

# 使用io.StringIO将字符串数据读入pandas DataFrame
df = pd.read_csv(io.StringIO(data), sep='\t', header=None)

# 设置列名 (数据集名称)
df.columns = ['Method', 'Dolly', 'Self-inst', 'Vicuna', 'Sinst']

# 将 "Method" 列设为索引
df = df.set_index('Method')

# 确保所有数值列都是数值类型
# 注意：在处理数据时，如果某个数值有小数点，确保它被正确解析为浮点数
df = df.apply(pd.to_numeric)

# 打印处理后的DataFrame，以便检查
print("处理后的数据:")
print(df)

# 设置Matplotlib的绘图风格为顶刊常用风格
plt.style.use('seaborn-v0_8-paper') # 一个简洁且适合发表的风格
# 或者可以自定义更细致的风格：
# plt.rcParams['font.family'] = 'serif' # 使用衬线字体，通常更专业
# plt.rcParams['font.serif'] = ['Times New Roman'] # 指定Times New Roman
# plt.rcParams['font.size'] = 12
# plt.rcParams['axes.labelsize'] = 14
# plt.rcParams['xtick.labelsize'] = 12
# plt.rcParams['ytick.labelsize'] = 12
# plt.rcParams['legend.fontsize'] = 12
# plt.rcParams['figure.dpi'] = 300 # 高分辨率输出

plt.figure(figsize=(10, 6)) # 设置图表大小

# 绘制折线图
# 使用marker='o'来在每个数据点显示圆点
# 使用linestyle='-'来确保是实线
# 可以通过colors参数自定义颜色列表
colors = plt.cm.get_cmap('tab10', len(df.index)) # 获取一组颜色，数量与方法数相同

for i, method in enumerate(df.index):
    plt.plot(df.columns, df.loc[method], marker='*', linestyle='-', label=method, color=colors(i))

# 添加标题和轴标签
plt.title('Performance Comparison Across Datasets and Methods', fontsize=16)
plt.xlabel('Dataset', fontsize=14)
plt.ylabel('RougeL', fontsize=14) # 根据您的实际指标填写，例如 "Accuracy", "F1 Score", "Loss" 等

# 添加网格线，使数据点更容易读取
plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例
plt.legend(title='Method', loc='best', borderaxespad=0.) # 将图例放在外面

# 调整布局，防止标签重叠
plt.tight_layout(rect=[0, 0, 0.85, 1]) # 调整右侧空间以容纳图例

# 显示图表
plt.show()

# 可以保存图表到文件，例如高质量的PNG或PDF
plt.savefig('/root/autodl-tmp/save/results/gpt2/image/medium.png', dpi=300, bbox_inches='tight')
# plt.savefig('performance_comparison_line_plot_updated.png', dpi=300, bbox_inches='tight')
# plt.savefig('performance_comparison_line_plot_updated.pdf', bbox_inches='tight')