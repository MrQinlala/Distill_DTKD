import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# Your updated data string from the "Loss set" table
data = """
Method	Dolly	Self-inst	Vicuna	Sinst
r"$w_1 \cdot FKL + w_2 \cdot RKL$ (AKL)"	22.55206	9.98032	15.35748	17.22866
r"$T_s^2 \cdot (w_1 \cdot FKL + w_2 \cdot RKL)$"	21.442	9.8172	14.7778	15.5044
r"$w_1 \cdot FKL + T_s^2 \cdot w_2 \cdot RKL$"	22.58874	10.17492	14.96796	17.10772
r"$T_s^2 \cdot w_1 \cdot FKL + w_2 \cdot RKL$ (ours)"	24.6742	11.729	15.7300	20.54
"""

# Use io.StringIO to read the string data into a pandas DataFrame
df = pd.read_csv(io.StringIO(data), sep='\t')

# Set "Method" column as index
df = df.set_index('Method')

# Ensure all value columns are numeric
df = df.apply(pd.to_numeric)

# Print processed DataFrame for verification
print("Processed Data:")
print(df)

# --- Matplotlib Plotting for Top Journal Style Bar Chart ---

# Set Matplotlib style for a professional look
plt.style.use('seaborn-v0_8-paper') # A clean, publication-ready style

# You can further customize font, size, etc. for more fine-grained control
# plt.rcParams['font.family'] = 'serif' # Use serif fonts, often preferred for academic papers
# plt.rcParams['font.serif'] = ['Times New Roman'] # Specify Times New Roman or other preferred serif font
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 300 # High resolution output for publication
plt.rcParams['text.usetex'] = True  # 使用 LaTeX 渲染
plt.rcParams['font.family'] = 'serif'  # 设置字体
plt.figure(figsize=(12, 7)) # Set figure size for better readability

# Data preparation for grouped bar chart
n_datasets = len(df.columns)
n_methods = len(df.index)
bar_width = 0.15 # Width of each individual bar
index = np.arange(n_datasets) # X-axis positions for each dataset group

# Generate a color map for different methods
custom_colors = ['#F6B7C6', '#A2DADE', '#D8CBF0', '#A2C2E2']

# colors = plt.cm.get_cmap('tab10', n_methods)

# Plotting the grouped bars
for i, method in enumerate(df.index):
    # Calculate position for each group of bars
    # This ensures bars for each method are grouped together for each dataset
    offset = bar_width * (i - (n_methods - 1) / 2)
    plt.bar(index + offset, df.loc[method], bar_width, label=method, color=custom_colors[i], edgecolor='black', linewidth=0.5)

# Add titles and labels
plt.title('Performance Comparison of Different Loss Function Setups', fontsize=16, pad=20)
plt.xlabel('Dataset', fontsize=14)
plt.ylabel('RougeL', fontsize=14) # Replace with your actual metric name (e.g., 'Loss', 'Accuracy')

# Set x-axis ticks and labels
plt.xticks(index, df.columns, fontsize=12)

# Add y-axis grid lines for easier reading
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add legend
# plt.legend(title='Loss Setup', loc='upper left', bbox_to_anchor=(0.99, 1), borderaxespad=0.) # Place legend outside
plt.legend(
    title='Loss Setup',
    loc='upper right',
    alignment='right',   # 控制图例文本右对齐
    prop={'size': 10},
    frameon=True,
    edgecolor='black',
    fancybox=False
)

# plt.legend(title='Loss Setup', loc='upper left', borderaxespad=0.) 
# Adjust layout to prevent labels from overlapping and ensure legend fits
plt.tight_layout() # Adjust right margin to make space for the legend

# Display the plot
# plt.show()

# You can save the plot to a high-quality file, e.g., PNG or PDF
plt.savefig('/root/autodl-tmp/save/results/gpt2/image/hes.png', dpi=300, bbox_inches='tight')
# plt.savefig('loss_setup_comparison_bar_chart.pdf', bbox_inches='tight')