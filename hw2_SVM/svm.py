from sklearn import svm
import numpy as np
import random
import matplotlib.pyplot as plt

X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)

def Create_cata(SIZE = 100,Num_in_cate = 20,par_x=0,par_y=0):
    # 创建一个样本
    # 边界大小为100*100
    a = int(SIZE / 2)
    rand_x_block = random.sample(range(a), Num_in_cate)
    rand_y_block = random.sample(range(a), Num_in_cate)
    category = []
    for i in range(Num_in_cate):
        category.append([rand_x_block[i]+par_x*a, rand_y_block[i]+par_y*a])
    # 创建一类中的数据集
    return category

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 10, x.max() + 10
    y_min, y_max = y.min() - 10, y.max() + 10
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

cata_a = Create_cata()
cata_b = Create_cata(par_x=1,par_y=1)
All_cata = cata_a+cata_b
label = [0]*20+[1]*20
clf = svm.SVC(kernel='linear')
clf.fit(All_cata, label)
# 拟合函数
nd = np.asarray(All_cata)
x = nd[:,0]
y = nd[:,1]
xx, yy = make_meshgrid(x, y)
fig, ax = plt.subplots()
plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(x[:20], y[:20], c='b', cmap=plt.cm.coolwarm, s=20, edgecolors='k',label='c1')
ax.scatter(x[20:], y[20:], c='r', cmap=plt.cm.coolwarm, s=20, edgecolors='k',label='c2')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('SVM result')
ax.legend()
plt.show()

# 添加离群点
outliner = [56,16]
outliner_l = 1
All_cata.append(outliner)
label.append(outliner_l)
clf.fit(All_cata, label)

plt.clf()
fig, ax = plt.subplots()
plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(x[:20], y[:20], c='b', cmap=plt.cm.coolwarm, s=20, edgecolors='k',label='c1')
ax.scatter(x[20:], y[20:], c='r', cmap=plt.cm.coolwarm, s=20, edgecolors='k',label='c2')
ax.scatter(outliner[0],outliner[1],c='g',cmap=plt.cm.coolwarm, s=20, edgecolors='k',label='outliner')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('SVM result with outliner')
ax.legend()
plt.show()