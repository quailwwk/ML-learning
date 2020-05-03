from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.datasets import load_iris

import numpy as np


class MyKmeans(BaseEstimator, TransformerMixin):
    def __init__(self, k=5, init_method='random', max_iter=100, random_state=42):
        self.k = k
        self.init_method = init_method
        self.max_iter = max_iter
        self.random_state = check_random_state(random_state)
        self.centers = None

    def init_centers(self, X, y=None, method=None):
        """
        初始化聚类中心
        :param X:array-like or sparse matrix, shape=(n_samples, n_features)
        :param y:
        :return: centers,(k, n_features)
        """
        if not method: method = self.init_method

        if method == 'random':
            init_indexs = self.random_state.randint(0, X.shape[0], size=self.k)
            return X[init_indexs]
        if method == 'kmeans++':
            pass

    def kmeans_plus(self, X, y=None):
        """
        kmeans++ 确定初始中心点
        :param X:
        :param y:
        :return:
        """
        c1_index = np.random.randint(0, X.shape[0])
        centers = [X[c1_index]]
        distances = np.full((X.shape[0], ), np.inf)
        min_distances = distances
        for _ in range(self.k - 1):
            distances = self.calcu_distance(X, centers[-1])  # 只计算上一步新加入的中心点的距离即可
            min_distances = np.minimum(min_distances, distances)
            prob = np.power(min_distances, 2) / np.power(min_distances, 2).sum()

    def RouletteWheel(self, prob):
        """
        轮盘法
        :param prob:
        :return:
        """



    def fit(self, X, y=None):
        """
        1. 在样本中随机选取![k]个样本点充当各个簇的初始中心点
        2. 计算所有样本点与各个簇中心之间的距离,然后把样本点划入最近的簇中
        3. 根据簇中已有的样本点，重新计算簇中心
        4. 重复2、3， 直至中心点不再发生变化或达到最大迭代次数
        """

        new_centers = self.init_centers(X)
        old_centers = new_centers + 1
        times = 0
        while not (new_centers == old_centers).all() and times <= self.max_iter:
            times += 1
            old_centers = new_centers
            new_centers = self.one_step(X, new_centers)
        self.centers = new_centers
        return self.centers

    def one_step(self, X, centers, y=None):
        """
        一步迭代
        :param X:
        :param centers: 此步迭代之前的中心点
        :return:
        """
        distance = np.zeros((X.shape[0], centers.shape[0]))

        for i, center in enumerate(centers):
            distance[:, i] = self.calcu_distance(X, center)

        res = np.argmin(distance, axis=1)

        for i in range(self.k):
            centers[i] = np.mean(X[res == i], axis=0)

        return centers

    def calcu_distance(self, X, center, method='Euclidean'):
        """
        计算样本与某个中心点间的距离
        :param method:
        :return:
        """
        if method == 'Euclidean': return np.sqrt(np.sum(np.power(X - center, 2), axis=1))


X = load_iris(return_X_y=True)[0]
print(MyKmeans(max_iter=10).kmeans_plus(X))
# print(MyKmeans(random_state=20).fit(X))
