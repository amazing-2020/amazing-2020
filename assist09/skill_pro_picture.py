import os
import numpy as np
from scipy import sparse
import networkx as nx
import matplotlib.pyplot as plt

# 加载数据
data_folder = ''
pro_skill_sparse = sparse.load_npz(os.path.join(data_folder, 'pro_skill_sparse.npz')).tocsc()

# 选择一个随机的技能
random_skill_index = np.random.randint(pro_skill_sparse.shape[1])

# 获取与随机选择技能相关的问题索引
related_problem_indices = pro_skill_sparse[:, random_skill_index].nonzero()[0]
# 限制每个技能最多添加10个相关问题
if len(related_problem_indices) > 10:
    related_problem_indices = np.random.choice(related_problem_indices, size=10, replace=False)

# 创建无向图
G = nx.Graph()

# 添加节点
for problem_index in related_problem_indices:
    G.add_node(problem_index)

# 添加边
for problem_index in related_problem_indices:
    for skill_index in pro_skill_sparse[problem_index].nonzero()[1]:
        G.add_edge(problem_index, skill_index)

# 绘制图形
pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels=True, font_weight='bold')
plt.show()
