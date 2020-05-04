from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

import numpy as np


class MyKmeans(BaseEstimator, TransformerMixin):
    """
    KMeans聚类，可实现基本聚类和Kmeans++聚类

    params:
    init_method: ‘random' or 'kmeans++', 初始点的取法
    n_init: int, 取初始点的次数，用于重复计算取最优结果
    max_iter: int, 最大迭代次数
    distance_method: 'Euclidean' or 'Manhattan', default='Euclidean'，距离的计算方式
    """

    def __init__(self, k=5, init_method='kmeans++', n_init=10, max_iter=300, distance_method='Manhattan', random_state=42):
        self.k = k
        self.init_method = init_method
        self.n_init = n_init
        self.max_iter = max_iter
        self.distance_method = distance_method
        self.random_state = check_random_state(random_state)
        self.cluster_centers_ = None
        self.labels_ = None
        self.score_ = None

    def init_centers(self, X, y=None, method=None):
        """
        初始化聚类中心
        :param X:array-like or sparse matrix, shape=(n_samples, n_features)
        :return: List[np.ndarray(k, n_features)], 每次训练的起始点列表
        """
        if not method: method = self.init_method

        if method == 'random':
            init_indexs = self.random_state.randint(0, X.shape[0], size=(self.n_init, self.k))
            return list(X[init_indexs])
        if method == 'kmeans++':
            # kmeans++只需指定一组初始点
            init_indexs = self.random_state.randint(0, X.shape[0], size=self.n_init)
            return self.kmeans_plus(X, init_indexs)

    def kmeans_plus(self, X, init_indexs, y=None):
        """
        kmeans++ 确定初始中心点
        :param X:
        :param y:
        :return:
        """
        c1_index = init_indexs
        res = []
        for i in range(self.n_init):
            centers = X[c1_index[i]]
            distances = np.full((X.shape[0],), np.inf)
            min_distances = distances
            for _ in range(self.k - 1):
                distances = self.calcu_distance(X, centers[-1])  # 只计算上一步新加入的中心点的距离即可
                min_distances = np.minimum(min_distances, distances)
                prob = np.power(min_distances, 2) / np.power(min_distances, 2).sum()
                centers = np.vstack((centers, X[self.RouletteWheel(prob)]))
            res.append(centers)

        return res

    def RouletteWheel(self, prob):
        """
        轮盘法
        :param prob:
        :return:
        """
        cum_prob = prob.cumsum()
        random_prob = np.random.random()
        return np.where(cum_prob >= random_prob)[0][0]

    def fit(self, X, y=None):
        """
        单次训练：
            1. 在样本中随机选取![k]个样本点充当各个簇的初始中心点
            2. 计算所有样本点与各个簇中心之间的距离,然后把样本点划入最近的簇中
            3. 根据簇中已有的样本点，重新计算簇中心
            4. 重复2、3， 直至中心点不再发生变化或达到最大迭代次数
        重复训练：
            用不同的初始点重复n_init次，取平方损失函数最小(score最大)的作为最终结果
        """
        best_score_ = -np.inf
        all_centers = self.init_centers(X)
        for i in range(self.n_init):
            new_centers = all_centers[i]
            # 保证初始的old_centers != new_centers
            old_centers = new_centers[:-1] + [new_centers[-1] + 1]
            times = 0
            while not all([np.allclose(x, y) for x, y in zip(old_centers, new_centers)]) and times <= self.max_iter:
                times += 1
                old_centers = new_centers
                new_centers, new_res = self.one_step(X, new_centers)
            score = self.score(X, centers=new_centers)
            if score >= best_score_:
                best_score_ = score
                best_centers_ = new_centers
                best_labels_ = self._transform(X, centers=new_centers)
        self.cluster_centers_ = best_centers_
        self.labels_ = best_labels_
        self.score_ = best_score_
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X)._transform(X)

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def _predict(self, X, y=None, centers=None):
        """
        聚类结果
        若未指定centers， 则必须先fit
        :return:
        """
        distances = self._transform(X, centers=centers)
        return np.argmin(distances, axis=1)

    def _transform(self, X, y=None, centers=None):
        """
        将输入数据转换为与各个聚类中心的距离
        若未指定centers， 则必须先fit
        :param X:
        :param y:
        :return: array, shape=(n_samples, self.k)
        """
        if centers is None: centers = self.cluster_centers_
        distances = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            distances[:, i] = self.calcu_distance(X, centers[i])
        return distances

    def score(self, X, y=None, centers=None):
        """
        最终得分，各个样本与聚类中心的距离的平方和
        若未指定centers，则需要先fit
        :return:
        """
        if centers is None: centers = self.cluster_centers_
        res = self._predict(X, centers=centers)
        score = 0
        for i, center in enumerate(centers):
            score -= np.power(self.calcu_distance(X[res == i], center), 2).sum()
        return score

    def one_step(self, X, centers, y=None):
        """
        一步迭代，将每个样本分入距离最近的类中，再计算每个类别的平均值作为新的聚类中心
        :param X:
        :param centers: 此步迭代之前的中心点
        :return: centers: 此步迭代之后的中心点，
                res: 此步聚类结果
        """
        distance = np.zeros((X.shape[0], len(centers)))

        for i, center in enumerate(centers):
            distance[:, i] = self.calcu_distance(X, center)

        res = np.argmin(distance, axis=1)

        for i in range(self.k):
            centers[i] = np.mean(X[res == i], axis=0)

        return centers, res

    def calcu_distance(self, X, center, method='Euclidean'):
        """
        计算样本与某个中心点间的距离
        :param method:
        :return:
        """
        if method == 'Euclidean': return np.sqrt(np.sum(np.power(X - center, 2), axis=1))
        if method == 'Manhattan': return np.sum(abs(X - center), axis=1)


X = load_iris(return_X_y=True)[0]
# print(MyKmeans().init_centers(X))
print(MyKmeans(max_iter=300, n_init=20).fit(X).score(X))
print(KMeans(n_clusters=5).fit(X).score(X))
