import numpy as np
import matplotlib.pyplot as plt

# 定义 Leaky ReLU 函数
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# 生成一组 x 值
x = np.linspace(-10, 10, 100)

# 计算对应的 y 值
y = leaky_relu(x)

# 绘制 Leaky ReLU 函数曲线
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('Leaky ReLU(x)')
plt.title('Leaky ReLU Function')
plt.grid(True)
plt.show()
