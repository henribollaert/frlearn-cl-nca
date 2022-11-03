import numpy as np
from abc import ABC

from algorithms.nca import NCA
from algorithms.lmnn import LMNN
from algorithms.dmlmj import DMLMJ

from .relations_base import DistanceFunction, DistanceFunctionFactory


class MahalanobisDistanceFunction(DistanceFunction):
    """
    Generic Mahalanobis pseudo-metric, which can be squared or not. It is always normalised.
    """

    __slots__ = ('matrix', 'max_d', 'squared', )

    def __init__(self, matrix, max_d, squared):
        self.matrix = matrix
        self.max_d = max_d
        self.squared = squared

    def __call__(self, a, b):
        d = np.matmul(np.matmul(np.transpose(a - b), self.matrix), (a - b))
        if not self.squared:
            d = np.sqrt(d)
        d /= self.max_d
        return d


class MahalanobisDistanceFactory(DistanceFunctionFactory, ABC):
    __slots__ = ('matrix', 'max_d', 'squared',)

    def __init__(self, squared=False):
        self.squared = squared

    def fit(self, X, y=None):
        self.max_d = 0
        for x1 in X:
            for x2 in X:
                d12 = np.matmul(np.matmul(np.transpose(x1 - x2), self.matrix), (x1 - x2))
                if not self.squared:
                    d12 = np.sqrt(d12)
                if d12 > self.max_d:
                    self.max_d = d12

    def get_metric(self) -> DistanceFunction:
        return MahalanobisDistanceFunction(self.matrix, self.max_d, self.squared)


class MahalanobisCorrelationDistanceFactory(MahalanobisDistanceFactory):
    """
    Factory that creates Mahalanobis distances with the inverse of the covariance matrix as defining matrix.
    It calculates the inverse of the covariance matrix of the data set
    and the maximum distance to return a normalised Mahalanobis distance.
    """

    def __init__(self, squared=False):
        super(MahalanobisCorrelationDistanceFactory, self).__init__(squared=squared)

    def fit(self, X, y=None):
        if self.can_apply(X):
            self.matrix = np.linalg.inv(np.cov(X, rowvar=False))
            super(MahalanobisCorrelationDistanceFactory, self).fit(X, y)

    def can_apply(self, X, y=None) -> bool:
        cov = np.cov(X, rowvar=False)
        return np.linalg.matrix_rank(cov) == cov.shape[0]

    @staticmethod
    def get_name():
        return "mah"


def neighbour_applier(y, k) -> bool:
    """
    Checks whether y contains at least k samples of each unique class.

    Parameters
    ----------
    y   Data set
    k   Minimum number of elements of each class.

    Returns
    -------
    True if y the above holds.
    """
    return k < np.bincount(np.unique(y, return_inverse=True)[1]).min(initial=np.infty)


class NCAFactory(MahalanobisDistanceFactory):
    """
    Factory which uses a de-cythonized version of pyDML's NCA implementation.
    It divides by the maximum distance.
    """

    __slots__ = ('model', 'matrix', 'squared', )

    def __init__(self, squared=False):
        super(NCAFactory, self).__init__(squared=squared)
        self.model = NCA()

    def fit(self, X, y=None):
        self.model.fit(X=X, y=y)
        self.matrix = self.model.metric()
        super(NCAFactory, self).fit(X, y)

    def can_apply(self, X, y=None) -> bool:
        return X.shape[0] < 4000

    @staticmethod
    def get_name():
        return "nca"


class LMNNFactory(MahalanobisDistanceFactory):
    """
    Factory which uses a de-cythonized version of pyDML's LMNN implementation.
    It divides by the maximum distance.
    """

    __slots__ = ('model', 'matrix', 'k', 'squared', )

    def __init__(self, k=3, squared=False):
        super(LMNNFactory, self).__init__(squared=squared)
        self.k = k
        self.model = LMNN(k=k)

    def fit(self, X, y=None):
        self.model.fit(X=X, y=y)
        self.matrix = self.model.metric()
        super(LMNNFactory, self).fit(X, y)

    def can_apply(self, X, y=None) -> bool:
        return X.shape[0] < 4000 and neighbour_applier(y, self.k)

    @staticmethod
    def get_name():
        return "lmnn"


class DMLMJFactory(MahalanobisDistanceFactory):
    """
    Factory which uses a de-cythonized version of pyDML's DMLMJ implementation.
    """
    __slots__ = ('model', 'matrix', 'k',)

    def __init__(self,
                 num_dims=None,
                 n_neighbors=3,
                 alpha=0.001,
                 reg_tol=1e-10,
                 squared=False):
        super(DMLMJFactory, self).__init__(squared=squared)
        self.k = n_neighbors
        self.model = DMLMJ(num_dims=num_dims,
                           n_neighbors=n_neighbors,
                           alpha=alpha,
                           reg_tol=reg_tol)

    def fit(self, X, y=None):
        self.model.fit(X=X, y=y)
        self.matrix = self.model.metric()
        super(DMLMJFactory, self).fit(X, y)

    def can_apply(self, X, y=None) -> bool:
        return X.shape[0] < 4000 and neighbour_applier(y, self.k)

    @staticmethod
    def get_name():
        return "dmlmj"


class ClassMahalanobisDistanceFunction(DistanceFunction):
    """
    Class specific Mahalanobis pseudo-metric that is normalised and can be squared or not.
    """

    __slots__ = ('matrix_dict', 'overall_matrix', 'overall_max_d', 'max_d_dict', 'X', 'y', 'squared', )

    def __init__(self, matrix_dict, overall_matrix, max_d_dict, overall_max_d, X, y, squared):
        self.matrix_dict = matrix_dict
        self.overall_matrix = overall_matrix
        self.max_d_dict = max_d_dict
        self.overall_max_d = overall_max_d
        self.X = X
        self.y = y
        self.squared = squared

    def __call__(self, a, b):
        m = self.overall_matrix
        d = self.overall_max_d
        # look for one of the samples in the training set and select its class
        ind_array = np.where(np.isclose(a, self.X).all(axis=1))[0]
        if len(ind_array) > 0:
            first_class = self.y[ind_array[0]]
            i = 1
            while i < len(ind_array) and self.y[ind_array[i]] == first_class:
                i += 1
            if i == len(ind_array):
                m = self.matrix_dict[first_class]
                d = self.max_d_dict[first_class]
        else:
            ind_array = np.where(np.isclose(b, self.X).all(axis=1))[0]
            if len(ind_array) > 0:
                first_class = self.y[ind_array[0]]
                i = 1
                while i < len(ind_array) and self.y[ind_array[i]] == first_class:
                    i += 1
                if i == len(ind_array):
                    m = self.matrix_dict[first_class]
                    d = self.max_d_dict[first_class]
        dist = np.matmul(np.matmul(np.transpose(a - b), m), (a - b))
        if not self.squared:
            dist = np.sqrt(dist)
        return dist/d


class ClassMahalanobisDistanceFactory(DistanceFunctionFactory):
    """
    Factory that creates fitted, class specific Mahalanobis distances, divided by max distance
    and with support for squared.
    """

    __slots__ = ('matrix_dict', 'general_matrix', 'overal_max_d', 'max_d_dict', 'X', 'y',)

    def __init__(self, squared=False):
        self.general_matrix = None
        self.overall_max_d = 0
        self.matrix_dict = None
        self.squared = squared

    def fit(self, X, y=None):
        if self.can_apply(X):
            # general matrix
            self.general_matrix = np.linalg.inv(np.cov(X, rowvar=False))
            # general max distance
            self.overall_max_d = 0
            for x1 in X:
                for x2 in X:
                    d12 = np.matmul(np.matmul(np.transpose(x1 - x2), self.general_matrix), (x1 - x2))
                    if not self.squared:
                        d12 = np.sqrt(d12)
                    if d12 > self.overal_max_d:
                        self.overall_max_d = d12

            # class specific matrix
            classes = np.unique(y)
            if self.matrix_dict is None:
                self.matrix_dict = dict()
                for c in classes:
                    self.matrix_dict[c] = self.general_matrix
                    c_index = np.where(y == c)[0]
                    # if there is only one element of this class, the covariance matrix is zero
                    if len(c_index) > 1:
                        c_x = X[c_index]
                        c_cov = np.cov(c_x, rowvar=False)
                        # can the covariance matrix be inverted
                        if np.linalg.matrix_rank(c_cov) == c_cov.shape[0]:
                            c_matrix = np.linalg.inv(c_cov)
                        else:
                            c_matrix = self.general_matrix
                        self.matrix_dict[c] = c_matrix
            # class specific max distance
            self.max_d_dict = dict()
            for c in classes:
                c_index = np.where(y == c)[0]
                c_x = X[c_index]
                c_matrix = self.matrix_dict[c]
                self.max_d_dict[c] = 0
                if len(c_index) > 1:
                    for x1 in c_x:
                        for x2 in c_x:
                            d12 = np.matmul(np.matmul(np.transpose(x1 - x2), c_matrix), (x1 - x2))
                            if not self.squared:
                                d12 = np.sqrt(d12)
                            if d12 > self.max_d_dict[c]:
                                self.max_d_dict[c] = d12

            # rest
            self.X = X
            self.y = y

    def get_metric(self) -> DistanceFunction:
        return ClassMahalanobisDistanceFunction(self.matrix_dict,
                                                self.general_matrix,
                                                self.max_d_dict,
                                                self.overal_max_d,
                                                self.X,
                                                self.y,
                                                self.squared)

    def can_apply(self, X, y=None) -> bool:
        cov = np.cov(X, rowvar=False)
        can = np.linalg.matrix_rank(cov) == cov.shape[0]
        i = 0
        classes = np.unique(y)
        while can and i < len(classes):
            c = classes[i]
            c_index = np.where(y == c)[0]
            c_x = X[c_index]
            if len(c_x) > 1:
                c_matrix = np.cov(c_x, rowvar=False)
                can = np.linalg.matrix_rank(c_matrix) == c_matrix.shape[0]
            i += 1
        return can

    @staticmethod
    def get_name():
        return "cl-mah"
