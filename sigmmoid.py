import numpy as np
import matplotlib.pyplot as plt

# 定义 Sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 生成一组 x 值
x = np.linspace(-10, 10, 100)

# 计算对应的 y 值
y = sigmoid(x)

# 绘制 Sigmoid 函数曲线
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
plt.title('Sigmoid Function')
plt.grid(True)
plt.show()
