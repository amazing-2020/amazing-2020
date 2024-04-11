import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 加载嵌入文件
embedding_data = np.load('bekt_dkt_model/pro_embed_bekt_dkt.npz')

# 获取问题和技能嵌入
# pro_repre = embedding_data['pro_repre']
#skill_repre_new = embedding_data['skill_repre']
pro_final_repre = embedding_data['pro_final_repre']

# 选择每个嵌入矩阵的第一列进行 t-SNE 降维
tsne = TSNE(n_components=2, random_state=42)
# pro_embeddings_2d = tsne.fit_transform(pro_repre[:, [0]])
#skill_embeddings_2d = tsne.fit_transform(skill_repre_new[:, [0]])
final_pro_embeddings_2d = tsne.fit_transform(pro_final_repre[:, [0]])

# 可视化
# plt.scatter(pro_embeddings_2d[:, 0], pro_embeddings_2d[:, 1], label='Problem Embeddings', marker='o')
#plt.scatter(skill_embeddings_2d[:, 0], skill_embeddings_2d[:, 1], label='Skill Embeddings', marker='x')
colors = np.arange(len(final_pro_embeddings_2d))
plt.scatter(final_pro_embeddings_2d[:, 0], final_pro_embeddings_2d[:, 1],c=colors, cmap='viridis', label='Final Problem Embeddings', marker='^')

# 添加标签或其他信息
# ...

plt.legend()
plt.title('t-SNE Visualization of Embeddings')
plt.show()
