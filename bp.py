import numpy as np
import matplotlib.pyplot as plt
import random

#生成x数据集
qq = []
for i in range(100):
    qq.append([random.randint(0,100),random.randint(0,100)])
qq = np.mat(qq).reshape(-1,2)
# print(qq)
z_labels = qq*[[3],[4]] + 5
# print(z_label)

num_y = 2
num_x = 2
k = np.mat([[1.,2.],[3.,4.]])

qq = np.mat(qq).reshape(-1,2)
b = np.mat([[1.],[2.]])
B = 0
w = np.ones((num_y,1))
w = np.mat(w)
α = 0.000001
result = []
test_x = []
num = 0
m,n = qq.shape
for index in range(m):
    for i in range(5):
        x = qq[index,:]
        z_label = z_labels[index,:]
        # print('x:',x)
        # print(z_label)
        y = k * x.T + b
        # print('y:',y)
        a = y
        a[0,0] = max(0,a[0,0])
        a[1,0] = max(0,a[1,0])
        z = w.T * a + B
        # print("误差:",np.square(z-z_label),z,z_label)
        result.append(np.square(z-z_label)[0,0])
        if np.square(z - z_label) < 0.0001:
            # print('收敛')
            break
        error = (z - z_label)
        w -= α * error[0, 0] * a
        B -= α * error[0, 0]
        k -= α*error[0,0] * (w * x).T
        b -= α*error[0,0] * w

final = []
for i in qq:
    y = k * i.T + b
    a = y
    a[0, 0] = max(0, a[0, 0])
    a[1, 0] = max(0, a[1, 0])
    mm = w.T * a + B
    final.append(mm[0,0])
# print(final)
xx = np.linspace(0,100,100)
plt.grid()
plt.plot(xx,z_labels,'r--')
plt.plot(xx,final,'g-')

plt.show()