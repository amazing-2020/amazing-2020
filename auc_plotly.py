import pandas as pd
import plotly.graph_objects as go

# 读取CSV文件
df = pd.read_csv('ednet/records.csv')

# 提取Epoch、Test AUC 列
epochs = df['Epoch']
test_auc = df['Test AUC']

# 使用Plotly绘制AUC曲线
fig = go.Figure()

fig.add_trace(go.Scatter(x=epochs, y=test_auc, mode='lines+markers', name='Test AUC'))

fig.update_layout(
    xaxis_title='Epochs',
    yaxis_title='Test AUC',
    title='Test AUC over Epochs',
    showlegend=True,
    # grid=dict(xside=True),
)

fig.show()
