import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse

# 加载数据
pro_skill_sparse = sparse.load_npz('pro_skill_sparse.npz').tocsc()
skill_skill_sparse = sparse.load_npz('skill_skill_sparse.npz').tocsc()

# 选择一个随机的技能
random_skill_index = np.random.randint(0, skill_skill_sparse.shape[0])

# 获取与该技能相关的其他技能索引
related_skill_indices = skill_skill_sparse[random_skill_index, :].nonzero()[1]

# 创建无向图
G = nx.Graph()



# 记录每个技能节点连接的问题节点数量
problem_count = {}

# 添加相关技能和其关联的问题到图中
for skill_index in related_skill_indices:
    # 获取与该技能相关的问题索引
    related_problem_indices = pro_skill_sparse[:, skill_index].nonzero()[0]
    # 随机选择最多 5 个与当前技能相关的问题节点
    np.random.shuffle(related_problem_indices)
    related_problem_indices = related_problem_indices[:5]
    # 添加技能节点
    G.add_node(skill_index, color='blue')
    # 添加问题节点，并记录连接的技能节点数量
    for problem_index in related_problem_indices:
        G.add_node(problem_index, color='red')
        # 添加技能与问题之间的边
        G.add_edge(skill_index, problem_index)
        if problem_index not in problem_count:
            problem_count[problem_index] = 1
        else:
            problem_count[problem_index] += 1

# 移除超过 5 个连接的问题节点
for problem_index, count in problem_count.items():
    if count > 5:
        G.remove_node(problem_index)

# 绘制图像
pos = nx.spring_layout(G, k=0.3)
node_colors = [G.nodes[n]['color'] for n in G.nodes]
nx.draw(G, pos, with_labels=True, node_color=node_colors, font_color='black', font_weight='bold')
plt.show()