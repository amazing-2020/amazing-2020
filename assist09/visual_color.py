import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 加载嵌入文件
embedding_data = np.load('embedding_200.npz')

# 获取最终问题嵌入
pro_final_repre = embedding_data['pro_final_repre']

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42)
final_pro_embeddings_2d = tsne.fit_transform(pro_final_repre)

# 创建一个连续的颜色映射，用于表示某一特征（这里使用默认的颜色映射）
colors = np.arange(len(final_pro_embeddings_2d))

# 可视化，根据某一特征上色
plt.scatter(final_pro_embeddings_2d[:, 0], final_pro_embeddings_2d[:, 1], c=colors, cmap='viridis', marker='^')

# 添加标签或其他信息
# ...

plt.colorbar()  # 添加颜色条
plt.title('t-SNE Visualization of Final Problem Embeddings with Coloring')
plt.show()
