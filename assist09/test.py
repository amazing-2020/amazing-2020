import os
import numpy as np
import pandas as pd
from scipy import sparse
import networkx as nx
import matplotlib.pyplot as plt

# 加载问题-技能稀疏矩阵
pro_skill_sparse = sparse.load_npz('pro_skill_sparse.npz').tocsc()

# 找到共享技能的问题
def find_shared_skill_problems(pro_skill_sparse, num_problems=3):
    shared_skill_problems = []
    while len(shared_skill_problems) < num_problems:
        # 随机选择一个问题
        problem_index = np.random.randint(0, pro_skill_sparse.shape[0])
        # 获取与该问题相关的技能索引
        related_skill_indices = pro_skill_sparse[problem_index, :].nonzero()[1]
        # 如果该问题与其他问题共享技能，则添加到列表中
        if len(set(related_skill_indices) & set(shared_skill_problems)) > 0:
            shared_skill_problems.append(problem_index)
        else:
            continue
    return shared_skill_problems

shared_skill_problems = find_shared_skill_problems(pro_skill_sparse)
print("Show1")
# 创建无向图
G = nx.Graph()
print("Show2")
# 添加边
for problem_index in shared_skill_problems:
    # 获取与问题相关的技能索引
    related_skill_indices = pro_skill_sparse[problem_index, :].nonzero()[1]
    # 随机选择最多10个相关技能
    random_skill_indices = np.random.choice(related_skill_indices, size=min(10, len(related_skill_indices)), replace=False)
    # 添加问题和相关技能之间的边
    for skill_index in random_skill_indices:
        G.add_edge(problem_index, skill_index)

# 绘制图形
print("Show3")
pos = nx.circular_layout(G)
print("Show4")
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=500, node_color='skyblue', edge_color='gray')
plt.title('Problems and Related Skills with Shared Skills')
plt.show()
