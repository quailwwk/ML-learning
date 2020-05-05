from scipy.stats import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np


class MykNN(BaseEstimator, TransformerMixin):
    """
    KNN分类算法
    k: int, default=5, 近邻的数量
    weights：str or callable， default='uniform',投票法的权重
            'uniform': 等权计算
            'inverse'：按照距离的倒数加权计算
            'Gaussian'：使用高斯函数
            [callable]: 人为指定
    metric：str，default='minkowski',距离度量
    p: int，default=2，minkowski距离的p值，p=2时即为欧氏距离
    """

    def __init__(self, k=5, weights='uniform', metric='minkowski', p=2):
        self.k = k
        self.weights = weights
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        distances = np.zeros((len(X), len(self.X_)))
        for i in range(len(X)):
            distances[i, :] = np.power(np.sum(np.power(X[i] - self.X_, self.p), axis=1), 1 / self.p)
        neighbors_ind = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        neighbors_labels = self.y_[neighbors_ind]

        if self.weights == 'uniform': return stats.mode(neighbors_labels).mode

        neighbors_distances = np.vstack((distances[i][neighbors_ind[i]] for i in range(len(X))))
        if self.weights == 'inverse':
            weights_ = 1 / neighbors_distances
        if self.weights == 'Gaussian':
            weights_ = np.exp(-np.square(neighbors_distances) / 2)
        return self.weighted_predict(neighbors_labels, weights_)

    def fit_predict(self, X, y, **fit_params):
        self.X_ = X
        self.y_ = y

        # 这里为了尽可能用np的向量化简化计算，对训练集自身的预测通过滚动训练集的方法进行。
        X_roll = X
        distances = np.zeros((len(X), len(X) - 1))
        for i in range(len(self.X_) - 1):
            X_roll = np.roll(X_roll, -1, axis=0)
            # 通过对样本整体滚动计算样本两两之间的距离
            distances[:, i] = np.power(np.sum(np.power(X - X_roll, self.p), axis=1), 1 / self.p)
        min_k_ind = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        # distances的第[i, j]个元素为x[i]与x[(i + j + 1) % len(X)]的距离，因此需要调整一下
        neighbors_ind = (min_k_ind + np.arange(1, len(X) + 1).reshape(-1, 1)) % len(X)
        neighbors_labels = y[neighbors_ind]

        # 等权预测
        if self.weights == 'uniform': return stats.mode(neighbors_labels, axis=1).mode.flatten()

        # 加权预测
        neighbors_distances = np.vstack((distances[i][min_k_ind[i]] for i in range(len(X))))
        # 距离倒数
        if self.weights == 'inverse':
            weights_ = 1 / neighbors_distances
        # 距离的高斯函数
        if self.weights == 'Gaussian':
            weights_ = np.exp(-np.square(neighbors_distances) / 2)
        return self.weighted_predict(neighbors_labels, weights_)

    def score(self, X, y):
        # TODO
        pass

    def weighted_predict(self, neighbors_labels, weights_):
        """
        加权对每个样本进行分类预测
        :param neighbors_labels: 所有样本的近邻们的标签， shape=(n_samples, k)
        :param weights_: 各个近邻的权重, shape=(n_samples, k)
        :return:
        """
        predict_ = np.zeros(neighbors_labels.shape[0])
        for i, sample in enumerate(neighbors_labels):
            best_predict = None
            best_weight = -np.inf
            for label in np.unique(sample):
                weight = weights_[i, sample == label].sum()
                if weight > best_weight: best_predict = label
            predict_[i] = best_predict
        return predict_


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
# print(MykNN(weights='uniform').fit_predict(X, y) - y)

print(MykNN(weights='Gaussian').fit(X_train, y_train).predict(X_test) - y_test)