import numpy as np

class KNNClassifier:

    def __init_(self, k):
        # 初始化kNN分类器
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None
        self.lables = None

    def fit(self, X_train, y_train, lables = None):
        #根据续联数据集X_train和y_train训练kNN分类器

        self._X_train = X_train
        self._y_train = y_train
        self._labels = lables
        return self

    def predict(self, X_predict):
        #给定待预测数据集X_predict, 返回标识X_predict的结果向量

        assert self._X_train is not None and self._y_train is not None, \
            "must fit before predict!"

        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)


    def _cal_weights(self, x):
        # 给定单个带预测数据x, 返回x的预测结果值
        assert x.shape[0] == self._X_train.shape[1], \
            "the feature number of x must be equal to X_train"

        # get vector distances, below both ok
        #distances = np.sqrt(np.sum((X_train-x)**2))
        distances = np.linalg.norm(self._X_train - x, axis=1)

        # arg sort distances
        nearest = np.argsort(distances)

        # get top k label
        topK_y = [self._y_train[i] for i in nearest[:self.k]]

        # cal result weights
        weights = np.bincount(np.array(topK_y))
        
        return weights
    
    def _predict(self, x):
        
        weights = self._cal_weights(x)
        return np.argmax(weights)    

    def show_predict(self, x):
        # 给定单个带预测数据x, 返回x的预测结果概率

        assert self._labels is not None, "lables must not be none"
        assert len(self._labels) > np.max(self._y_train), \
            "labels size must bigger then y_train range!"

        weights = self._cal_weights(x)
        self.print_weights(weights, self._labels)
        return self
    

    def print_weights(self, weights, labels = None):

        total = sum(weights)

        if labels is not None:
            assert len(weights) == len(labels), "the size of weights must equal to the size of labels"

            for i, val in enumerate(weights):
                print("[%s]概率:[%s]" % (labels[i], val/total))

        else:
            for i, val in enumerate(weights):
                print("[%s]概率:[%s]" % (i, val/total))
