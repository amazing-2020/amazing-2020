import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 加载嵌入数据
embedding_data = np.load('embedding.npz')

# 提取问题和技能的嵌入
pro_embed = embedding_data['pro_embed']
skill_embed = embedding_data['skill_embed']

# 将问题和技能嵌入合并为一个数组
all_embed = np.concatenate((pro_embed, skill_embed), axis=0)

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embed_tsne = tsne.fit_transform(all_embed)

# 绘制散点图
plt.figure(figsize=(10, 8))

# 提取问题和技能的 t-SNE 降维结果
pro_embed_tsne = embed_tsne[:pro_embed.shape[0]]
skill_embed_tsne = embed_tsne[pro_embed.shape[0]:]

# 绘制问题嵌入
plt.scatter(pro_embed_tsne[:, 0], pro_embed_tsne[:, 1], label='Problem Embedding', alpha=0.5)

# 绘制技能嵌入
plt.scatter(skill_embed_tsne[:, 0], skill_embed_tsne[:, 1], label='Skill Embedding', alpha=0.5)

# 添加图例和标题
plt.legend()
plt.title('t-SNE Visualization of Problem and Skill Embeddings')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# 显示图形
plt.show()
