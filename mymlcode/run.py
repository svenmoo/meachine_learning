import numpy as np
from lib import kNN_func as kfunc 

np.random.seed(100)
train = np.random.normal(5, 3, 100).reshape(-1, 2)
#train

dest = (np.sum(train, axis=1) > 8) + 0
labels = ["绿色", "红色"]

k = 6
v1 = np.array([2,4])
res = kfunc.kNN_func(k, train, dest, v1, labels)

#add
print(res)  

