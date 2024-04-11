import os
import numpy as np
from scipy import sparse
import networkx as nx
import matplotlib.pyplot as plt

# 加载问题-技能稀疏矩阵
pro_skill_sparse = sparse.load_npz('pro_skill_sparse.npz').tocsc()

# 随机选择一个技能
random_skill_index = np.random.randint(0, pro_skill_sparse.shape[1])

# 获取与所选技能相关的问题索引
related_problem_indices = pro_skill_sparse[:, random_skill_index].nonzero()[0]

# 创建无向图
G = nx.Graph()

# 添加问题节点
for i, problem_index in enumerate(related_problem_indices):
    G.add_node(f'Problem {problem_index}')

# 添加技能节点
G.add_node(f'Skill {random_skill_index}')

# 添加问题和技能之间的边
for problem_index in related_problem_indices:
    G.add_edge(f'Problem {problem_index}', f'Skill {random_skill_index}')

# 绘制图形
plt.figure(figsize=(10, 6))

# 使用spring_layout布局
pos = nx.spring_layout(G, k=0.2, seed=42)

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=200, node_color='skyblue')

# 绘制边
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

# 绘制节点标签
nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

# 添加标题
plt.title('Skill-Problem Relationship Graph')

# 显示图形
plt.axis('off')
plt.show()
