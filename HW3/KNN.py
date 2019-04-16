# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[2.2, 3.4], [3.9, 2.9], [3.7, 3.6], [4, 4], [2.8, 3.5], [3.5, 1], [3.8, 4], [3.1, 2.5]])
y = np.array([45, 55, 91, 142, 88, 2600, 163, 67])

X1 = np.array([[3.5, 3.6]])

k = 3
clf = KNeighborsClassifier(n_neighbors = k, p = 2)
clf.fit(X, y)
z = clf.predict(X1)
neighbors = clf.kneighbors(X1, return_distance = False)

tt = 0
for i in neighbors[0]:
    tt += y[i]
print(tt/3)


fig, ax = plt.subplots(2, 1, figsize = (10, 10))
ax[0].scatter(X[:, 0], X[:, 1], c = 'red', marker = '*')
ax[0].plot(X1[0, 0], X1[0, 1], c = 'green', marker = '^')

ax[1].scatter(X[:, 0], X[:, 1], c = 'red', marker = '*')
ax[1].plot(X1[0, 0], X1[0, 1], c = 'green', marker = '^')
for i in neighbors[0]:
    ax[1].plot([X[i][0], X1[0][0]], [X[i][1], X1[0][1]], 'k--', linewidth = 0.6)
plt.show()

