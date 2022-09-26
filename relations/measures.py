import numpy as np
from frlearn.vector_size_measures import MinkowskiSize
from abc import ABC, abstractmethod
from frlearn.uncategorised.weights import LinearWeights
from frlearn.neighbour_search_methods import BallTree
from frlearn.classifiers import FRNN
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score
from frlearn.base import select_class
from algorithms.clnca import ClassNCA
from algorithms.nca import NCA
from algorithms.lmnn import LMNN
from algorithms.dmlmj import DMLMJ


def get_new_factories():
    factories = [ManhattanDistanceFactory(),
                 EuclideanDistanceFactory(),
                 ChebyshevDistanceFactory(),
                 CosineMeasureFactory(),
                 CanberraFactory(),
                 MahalanobisDistanceFactory()]
    return factories


class DistanceFunction(ABC):
    """
    Abstract base class for distances.
    """

    @abstractmethod
    def __call__(self, a, b):
        pass


class DistanceFunctionFactory(ABC):
    """
    Abstract base class for the factories that return distances fitted to the data.
    """

    def can_apply(self, X, y=None) -> bool:
        """
        Check whether we can apply this distance function to the decision system given by (X,y).
        Parameters
        ----------
        X conditional attribute values for the samples
        y classes for the samples

        Returns
        -------
        True if we can apply the distance function, False otherwise
        """
        return True

    def fit(self, X, y=None):
        pass

    @abstractmethod
    def get_metric(self) -> DistanceFunction:
        pass


class ManhattanDistanceFactory(DistanceFunctionFactory):
    """
    Factory that returns a wrapper for frlearn's Manhattan distance
    """

    def get_metric(self) -> DistanceFunction:
        return ManhattanDistanceFactory.ManhattanDistance()

    @staticmethod
    def get_name():
        return "man"

    class ManhattanDistance(DistanceFunction):
        __slots__ = ('dist',)

        def __init__(self):
            self.dist = MinkowskiSize(p=1, unrooted=False, scale_by_dimensionality=True)

        def __call__(self, a, b):
            return self.dist(b - a)


class EuclideanDistanceFactory(DistanceFunctionFactory):
    """
    Factory that returns a wrapper for frlearn's Euclidean distance
    """

    def get_metric(self) -> DistanceFunction:
        return EuclideanDistanceFactory.EuclideanDistance()

    @staticmethod
    def get_name():
        return "euc"

    class EuclideanDistance(DistanceFunction):
        __slots__ = ('dist',)

        def __init__(self):
            self.dist = MinkowskiSize(p=2, unrooted=False, scale_by_dimensionality=True)

        def __call__(self, a, b):
            return self.dist(b - a)


class ChebyshevDistanceFactory(DistanceFunctionFactory):
    """
    Factory that returns a wrapper for frlearn's Chebyshev distance
    """

    def get_metric(self) -> DistanceFunction:
        return ChebyshevDistanceFactory.ChebyshevDistance()

    @staticmethod
    def get_name():
        return "che2"

    class ChebyshevDistance(DistanceFunction):
        __slots__ = ('dist',)

        def __init__(self):
            self.dist = MinkowskiSize(p=np.inf, unrooted=False, scale_by_dimensionality=True)

        def __call__(self, a, b):
            return self.dist(b - a)


class CorrelationFactory(DistanceFunctionFactory):
    """
    Factory that returns the fraction distance derived from the correlation.
    """

    __slots__ = ('averages',)

    def fit(self, X, y=None):
        self.averages = np.mean(X, axis=0)

    def get_metric(self) -> DistanceFunction:
        return CorrelationFactory.CorrelationFractionDistance(self.averages)

    @staticmethod
    def get_name():
        return "cor"

    class CorrelationFractionDistance(DistanceFunction):
        """
        Correlation distance based on fraction conversion.
        """

        def __init__(self, averages):
            self.averages = averages

        def __call__(self, a, b):
            nom = np.sum(np.multiply(a - self.averages, b - self.averages))
            den = np.sqrt(np.sum(np.square(a - self.averages)) * np.sum(np.square(b - self.averages)))
            return (1 - nom / den) / 2


class CosineMeasureFactory(DistanceFunctionFactory):
    """
    Factory that returns the classical normalised distance derived from the cosine similarity measure.
    """

    def __init__(self, normalised=True):
        self.normalised = normalised

    def get_metric(self) -> DistanceFunction:
        return CosineMeasureFactory.CosineDistance(self.normalised)

    def can_apply(self, X, y=None) -> bool:
        return not (X == 0).all(axis=1).any()

    @staticmethod
    def get_name():
        name = "cos"
        # if normalised:
        return "norm-cos"

    class CosineDistance(DistanceFunction):
        def __init__(self, normalised):
            self.normalised = normalised

        def __call__(self, a, b):
            return (1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))/2


class CanberraFactory(DistanceFunctionFactory):
    """
    Factory that returns the normalised canberra distance.
    """

    __slots__ = ("dimension", )

    def fit(self, X, y=None):
        self.dimension = X.shape[1]

    def get_metric(self) -> DistanceFunction:
        return CanberraFactory.CanberraDistance(self.dimension)

    @staticmethod
    def get_name():
        return "can"

    class CanberraDistance(DistanceFunction):
        __slots__ = ("dimension", )

        def __init__(self, dimension):
            self.dimension = dimension

        def __call__(self, a, b):
            nom = np.abs(a - b)
            den = np.abs(a) + np.abs(b)
            #  in the canberra metric, we set 0/0 to be 0
            return np.sum(np.divide(nom, den, out=np.zeros_like(nom), where=den != 0))/self.dimension

class MahalanobisDistanceFunction(DistanceFunction):
    """
    Generic Mahalanobis pseudo-metric, which can be squared or not.
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


class MahalanobisDistanceFactory(DistanceFunctionFactory):
    """
    Factory that creates Mahalanobis distances. It
    """

    __slots__ = ('matrix', 'max_d', 'squared', )

    def __init__(self, matrix=None, squared=False):
        self.squared = squared
        self.matrix = matrix

    def fit(self, X, y=None):
        if self.can_apply(X):
            if self.matrix is None:
                self.matrix = np.linalg.inv(np.cov(X, rowvar=False))
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

    def can_apply(self, X, y=None) -> bool:
        if self.matrix is not None:
            return True
        cov = np.cov(X, rowvar=False)
        return np.linalg.matrix_rank(cov) == cov.shape[0]

    # todo how to convey square here?
    @staticmethod
    def get_name():
        return "mah"


def neighbour_applier(y, k) -> bool:
    return k < np.bincount(np.unique(y, return_inverse=True)[1]).min(initial=np.infty)


class COMBOFactory(DistanceFunctionFactory):
    """
    Factory that selects the best distance for a given training set using cross-validation.
    """

    __slots__ = ('distances', 'folds', 'fitted_factory', 'k', 'weights', 'verbose',)

    def __init__(self,
                 distances=None,
                 folds=5,
                 k=20,
                 weights=LinearWeights(),
                 verbose=False
                 ):
        if distances is None:
            distances = get_new_factories()
        self.distances = distances
        self.folds = folds
        self.k = k
        self.weights = weights
        self.verbose = verbose

    def fit(self, X, y=None):
        distances = self.distances.copy()
        accuracies = {f: [] for f in distances}
        kf = KFold(n_splits=self.folds, shuffle=True, random_state=0)
        for train_index, test_index in kf.split(X, y):
            # get the train and test sets of this fold
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # remove any distances which are not defined from further consideration
            distances = [m for m in distances if m.can_apply(x_train, y_train) and m.can_apply(x_test, y_test)]

            # calculate the accuracies for FRNN with each measure on this fold
            for measure in distances:
                # fit the measure to the training set
                # print(measure.get_name())
                measure.fit(x_train, y_train)
                # instantiate the FRNN classifier factory
                clf = FRNN(preprocessors=(),
                           nn_search=BallTree(),
                           dissimilarity=measure.get_metric(),
                           lower_k=self.k,
                           upper_k=self.k,
                           lower_weights=self.weights,
                           upper_weights=self.weights)
                # construct the model
                model = clf(x_train, y_train)
                # query on the test set
                scores = model(x_test)
                # select classes with the highest scores and calculate the accuracy.
                classes = select_class(scores, labels=model.classes)
                accuracies[measure].append(balanced_accuracy_score(y_test, classes))

        # select the distance with the best average accuracy which never failed
        avg_accuracies = {f: sum(accuracies[f]) / self.folds for f in distances}
        best = distances[0]
        for f in distances[1:]:
            if avg_accuracies[f] > avg_accuracies[best]:
                best = f
        if self.verbose:
            print(best.get_name())
        # part of the fit function, thus we need to return it fitted to the training set
        best.fit(X, y)
        self.fitted_factory = best

    def get_metric(self) -> DistanceFunction:
        return self.fitted_factory.get_metric()

    @staticmethod
    def get_name():
        return "combo3"


class NCAFactory(MahalanobisDistanceFactory):
    """
    Factory which uses a de-cythonized version of pyDML's NCA implementation.
    It divides by the maximum distance.
    """

    __slots__ = ('model', 'matrix',)

    def __init__(self, squared=False):
        super().__init__(squared=squared)
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


class KernelDistance(DistanceFunction):
    """
    Generic distance function such that d(x,y) = 1 - R(x,y), where R(x,y) is derived from a kernel k,
    which is set at object creation.
    """

    __slots__ = ('kernel',)

    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, a, b):
        return 1 - self.kernel(a, b)


class GaussianKernelFactory(DistanceFunctionFactory):
    """
    Factory for distance based on a relation based on the Gaussian kernel.
    """

    __slots__ = ('gamma', 'folds', 'k', 'weights', 'verbose')

    def __init__(self, gamma=1, folds=0, k=3, weights=LinearWeights(), verbose=False):
        self.gamma = gamma
        self.folds = folds
        self.k = k
        self.weights = weights
        self.verbose = verbose

    def get_metric(self) -> DistanceFunction:
        return KernelDistance(lambda a, b: np.exp(-1 * np.linalg.norm(a - b) ** 2 / self.gamma))

    @staticmethod
    def get_name():
        return "gauss"


possible_gammas = [0.0001, 0.001, 0.1, 1, 10, 100]


class CVGaussianKernelFactory(DistanceFunctionFactory):
    """
    Factory for distance based on a relation based on the Gaussian kernel.
    """

    __slots__ = ('gamma', 'folds', 'k', 'weights', 'verbose')

    def __init__(self, gamma=1, folds=10, k=3, weights=LinearWeights(), verbose=False):
        self.gamma = gamma
        self.folds = folds
        self.k = k
        self.weights = weights
        self.verbose = verbose

    def fit(self, X, y=None):
        """
        Choosing the best gamma value for the dataset (X,y) using cross-validation, if folds was > 0 at initialisation.
        Parameters
        ----------
        X: samples
        y: class labels

        Returns nothing
        -------

        """
        accuracies = {g: [] for g in possible_gammas}
        kf = KFold(n_splits=self.folds, shuffle=True, random_state=0)
        for train_index, test_index in kf.split(X, y):
            # get the train and test sets of this fold
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # calculate the accuracies for FRNN with each measure on this fold
            for g in possible_gammas:
                # instantiate the FRNN classifier factory
                clf = FRNN(preprocessors=(),
                           nn_search=BallTree(),
                           dissimilarity=KernelDistance(lambda a, b: np.exp(-1 * np.linalg.norm(a - b) ** 2 / g)),
                           lower_k=self.k,
                           upper_k=self.k,
                           lower_weights=self.weights,
                           upper_weights=self.weights)
                # construct the model
                model = clf(x_train, y_train)
                # query on the test set
                scores = model(x_test)
                # select classes with the highest scores and calculate the accuracy.
                classes = select_class(scores, labels=model.classes)
                accuracies[g].append(balanced_accuracy_score(y_test, classes))

        # select the distance with the best average accuracy which never failed
        avg_accuracies = {g: sum(accuracies[g]) / self.folds for g in possible_gammas}
        best = possible_gammas[0]
        for g in possible_gammas[1:]:
            if avg_accuracies[g] > avg_accuracies[best]:
                best = g
        if self.verbose:
            print(best)
        self.gamma = best

    def get_metric(self) -> DistanceFunction:
        return KernelDistance(lambda a, b: np.exp(-1 * np.linalg.norm(a - b) ** 2 / self.gamma))

    @staticmethod
    def get_name():
        return "gauss-cv"


class ExponentialKernelFactory(DistanceFunctionFactory):
    """
    Factory for distance based on a relation based on the exponential kernel.
    """

    def __init__(self, gamma=1):
        self.gamma = gamma

    def get_metric(self) -> DistanceFunction:
        return KernelDistance(lambda a, b: np.exp(-1 * np.linalg.norm(a - b) / self.gamma))

    @staticmethod
    def get_name():
        return "exponential"


class CVExponentialKernelFactory(DistanceFunctionFactory):
    """
    Factory for distance based on a relation based on the Gaussian kernel.
    """

    __slots__ = ('gamma', 'folds', 'k', 'weights', 'verbose')

    def __init__(self, gamma=1, folds=10, k=3, weights=LinearWeights(), verbose=False):
        self.gamma = gamma
        self.folds = folds
        self.k = k
        self.weights = weights
        self.verbose = verbose

    def fit(self, X, y=None):
        """
        Choosing the best gamma value for the dataset (X,y) using cross-validation, if folds was > 0 at initialisation.
        Parameters
        ----------
        X: samples
        y: class labels

        Returns nothing
        -------

        """
        accuracies = {g: [] for g in possible_gammas}
        kf = KFold(n_splits=self.folds, shuffle=True, random_state=0)
        for train_index, test_index in kf.split(X, y):
            # get the train and test sets of this fold
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # calculate the accuracies for FRNN with each measure on this fold
            for g in possible_gammas:
                # instantiate the FRNN classifier factory
                clf = FRNN(preprocessors=(),
                           nn_search=BallTree(),
                           dissimilarity=KernelDistance(lambda a, b: np.exp(-1 * np.linalg.norm(a - b) / g)),
                           lower_k=self.k,
                           upper_k=self.k,
                           lower_weights=self.weights,
                           upper_weights=self.weights)
                # construct the model
                model = clf(x_train, y_train)
                # query on the test set
                scores = model(x_test)
                # select classes with the highest scores and calculate the accuracy.
                classes = select_class(scores, labels=model.classes)
                accuracies[g].append(balanced_accuracy_score(y_test, classes))

        # select the distance with the best average accuracy which never failed
        avg_accuracies = {g: sum(accuracies[g]) / self.folds for g in possible_gammas}
        best = possible_gammas[0]
        for g in possible_gammas[1:]:
            if avg_accuracies[g] > avg_accuracies[best]:
                best = g
        if self.verbose:
            print(best)
        self.gamma = best

    def get_metric(self) -> DistanceFunction:
        return KernelDistance(lambda a, b: np.exp(-1 * np.linalg.norm(a - b) / self.gamma))

    @staticmethod
    def get_name():
        return "exponential-cv"


class RationalQuadraticKernelFactory(DistanceFunctionFactory):
    """
    Factory for distance based on a relation based on the rational quadratic kernel.
    """

    def __init__(self, gamma=1):
        self.gamma = gamma

    def get_metric(self) -> DistanceFunction:
        return KernelDistance(lambda a, b: self.gamma / (np.square(np.linalg.norm(a - b)) + self.gamma))

    @staticmethod
    def get_name():
        return "rat-qua"


class CVRationalQuadraticKernelFactory(DistanceFunctionFactory):
    """
    Factory for distance based on a relation based on the Gaussian kernel.
    """

    __slots__ = ('gamma', 'folds', 'k', 'weights', 'verbose')

    def __init__(self, gamma=1, folds=10, k=3, weights=LinearWeights(), verbose=False):
        self.gamma = gamma
        self.folds = folds
        self.k = k
        self.weights = weights
        self.verbose = verbose

    def fit(self, X, y=None):
        """
        Choosing the best gamma value for the dataset (X,y) using cross-validation, if folds was > 0 at initialisation.
        Parameters
        ----------
        X: samples
        y: class labels

        Returns nothing
        -------

        """
        accuracies = {g: [] for g in possible_gammas}
        kf = KFold(n_splits=self.folds, shuffle=True, random_state=0)
        for train_index, test_index in kf.split(X, y):
            # get the train and test sets of this fold
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # calculate the accuracies for FRNN with each measure on this fold
            for g in possible_gammas:
                # instantiate the FRNN classifier factory
                clf = FRNN(preprocessors=(),
                           nn_search=BallTree(),
                           dissimilarity=KernelDistance(lambda a, b: self.gamma / (np.square(np.linalg.norm(a - b)) + g)),
                           lower_k=self.k,
                           upper_k=self.k,
                           lower_weights=self.weights,
                           upper_weights=self.weights)
                # construct the model
                model = clf(x_train, y_train)
                # query on the test set
                scores = model(x_test)
                # select classes with the highest scores and calculate the accuracy.
                classes = select_class(scores, labels=model.classes)
                accuracies[g].append(balanced_accuracy_score(y_test, classes))

        # select the distance with the best average accuracy which never failed
        avg_accuracies = {g: sum(accuracies[g]) / self.folds for g in possible_gammas}
        best = possible_gammas[0]
        for g in possible_gammas[1:]:
            if avg_accuracies[g] > avg_accuracies[best]:
                best = g
        if self.verbose:
            print(best)
        self.gamma = best

    def get_metric(self) -> DistanceFunction:
        return KernelDistance(lambda a, b: self.gamma / (np.square(np.linalg.norm(a - b)) + self.gamma))

    @staticmethod
    def get_name():
        return "rat-cv"


def circular_kernel(a, b, g):
    if np.linalg.norm(a - b) < g:
        t1 = 2 / np.pi * np.arccos(np.linalg.norm(a - b) / g)
        t2 = 2 / np.pi * np.linalg.norm(a - b) / g * np.sqrt(1 - np.square(np.linalg.norm(a - b) / g))
        return t1 - t2
    return 0


class CircularKernelFactory(DistanceFunctionFactory):
    """
    Factory for distance based on a relation based on the Gaussian kernel.
    """

    def __init__(self, gamma=1):
        self.gamma = gamma

    def get_metric(self) -> DistanceFunction:
        return KernelDistance(lambda a, b: circular_kernel(a, b, self.gamma))

    @staticmethod
    def get_name():
        return "circle"


class CVCircularKernelFactory(DistanceFunctionFactory):
    """
    Factory for distance based on a relation based on the Gaussian kernel.
    """

    __slots__ = ('gamma', 'folds', 'k', 'weights', 'verbose')

    def __init__(self, gamma=1, folds=10, k=3, weights=LinearWeights(), verbose=False):
        self.gamma = gamma
        self.folds = folds
        self.k = k
        self.weights = weights
        self.verbose = verbose

    def fit(self, X, y=None):
        """
        Choosing the best gamma value for the dataset (X,y) using cross-validation, if folds was > 0 at initialisation.
        Parameters
        ----------
        X: samples
        y: class labels

        Returns nothing
        -------

        """
        accuracies = {g: [] for g in possible_gammas}
        kf = KFold(n_splits=self.folds, shuffle=True, random_state=0)
        for train_index, test_index in kf.split(X, y):
            # get the train and test sets of this fold
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # calculate the accuracies for FRNN with each measure on this fold
            for g in possible_gammas:
                # instantiate the FRNN classifier factory
                clf = FRNN(preprocessors=(),
                           nn_search=BallTree(),
                           dissimilarity=KernelDistance(lambda a, b: circular_kernel(a, b, g)),
                           lower_k=self.k,
                           upper_k=self.k,
                           lower_weights=self.weights,
                           upper_weights=self.weights)
                # construct the model
                model = clf(x_train, y_train)
                # query on the test set
                scores = model(x_test)
                # select classes with the highest scores and calculate the accuracy.
                classes = select_class(scores, labels=model.classes)
                accuracies[g].append(balanced_accuracy_score(y_test, classes))

        # select the distance with the best average accuracy which never failed
        avg_accuracies = {g: sum(accuracies[g]) / self.folds for g in possible_gammas}
        best = possible_gammas[0]
        for g in possible_gammas[1:]:
            if avg_accuracies[g] > avg_accuracies[best]:
                best = g
        if self.verbose:
            print(best)
        self.gamma = best

    def get_metric(self) -> DistanceFunction:
        return KernelDistance(lambda a, b: circular_kernel(a, b, self.gamma))

    @staticmethod
    def get_name():
        return "circle-cv"


def spherical_kernel(a, b, g):
    if np.linalg.norm(a - b) < g:
        t1 = 3 / 2 * np.linalg.norm(a - b) / g
        t2 = 1 / 2 * np.power(np.linalg.norm(a - b) / g, 3)
        return 1 - t1 + t2
    return 0


class SphericalKernelFactory(DistanceFunctionFactory):
    """
    Factory for distance based on a relation based on the Gaussian kernel.
    """

    def __init__(self, gamma=1):
        self.gamma = gamma

    def get_metric(self) -> DistanceFunction:
        return KernelDistance(lambda a, b: spherical_kernel(a, b, self.gamma))

    @staticmethod
    def get_name():
        return "sphere-fixed"


class CVSphericalKernelFactory(DistanceFunctionFactory):
    """
    Factory for distance based on a relation based on the Gaussian kernel.
    """

    __slots__ = ('gamma', 'folds', 'k', 'weights', 'verbose')

    def __init__(self, gamma=1, folds=10, k=3, weights=LinearWeights(), verbose=False):
        self.gamma = gamma
        self.folds = folds
        self.k = k
        self.weights = weights
        self.verbose = verbose

    def fit(self, X, y=None):
        """
        Choosing the best gamma value for the dataset (X,y) using cross-validation, if folds was > 0 at initialisation.
        Parameters
        ----------
        X: samples
        y: class labels

        Returns nothing
        -------

        """
        accuracies = {g: [] for g in possible_gammas}
        kf = KFold(n_splits=self.folds, shuffle=True, random_state=0)
        for train_index, test_index in kf.split(X, y):
            # get the train and test sets of this fold
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # calculate the accuracies for FRNN with each measure on this fold
            for g in possible_gammas:
                # instantiate the FRNN classifier factory
                clf = FRNN(preprocessors=(),
                           nn_search=BallTree(),
                           dissimilarity=KernelDistance(lambda a, b: spherical_kernel(a, b, g)),
                           lower_k=self.k,
                           upper_k=self.k,
                           lower_weights=self.weights,
                           upper_weights=self.weights)
                # construct the model
                model = clf(x_train, y_train)
                # query on the test set
                scores = model(x_test)
                # select classes with the highest scores and calculate the accuracy.
                classes = select_class(scores, labels=model.classes)
                accuracies[g].append(balanced_accuracy_score(y_test, classes))

        # select the distance with the best average accuracy which never failed
        avg_accuracies = {g: sum(accuracies[g]) / self.folds for g in possible_gammas}
        best = possible_gammas[0]
        for g in possible_gammas[1:]:
            if avg_accuracies[g] > avg_accuracies[best]:
                best = g
        if self.verbose:
            print(best)
        self.gamma = best

    def get_metric(self) -> DistanceFunction:
        return KernelDistance(lambda a, b: spherical_kernel(a, b, self.gamma))

    @staticmethod
    def get_name():
        return "sphere-cv"


class GradientKernelFactory(DistanceFunctionFactory, ABC):
    """
    Abstract base class for factories for distances based on a relation based on a kernel,
    which is fit to the data using gradient descent.
    """

    # __slots__ = ('gamma', 'batch_size', 'k', 'weights', 'verbose', 'X', 'y', 'n_samples', 'n_features',
    #              'learning_rate', 'max_its', 'precision')

    def __init__(self,
                 kernel,
                 gradient,
                 gamma=1,
                 batch_size=10,
                 k=3,
                 weights=LinearWeights(),
                 learning_rate=0.01,
                 max_its=10000,
                 precision=0.00001,
                 verbose=False):
        """

        Parameters
        ----------
        kernel          lambda (a,b, gamma) -> kernel_gamma(a,b)
        gradient        lambda (a,b, gamma) -> gradient_gamma(a,b)
        gamma           initial value for gamma
        batch_size      number of gradients to calculate in each step
        k               number of neighbours used in FRNN
        weights         weights used in FRNN
        learning_rate   represents the size of the steps taken in gradient descent
        max_its         maximum number of iterations
        precision       minimum difference in gamma between two subsequent iterations
        verbose         print stuff or not
        """
        self.kernel = kernel
        self.gradient = gradient
        self.initial_gamma = gamma
        self.gamma = gamma
        self.batch_size = batch_size
        self.k = k
        self.weights = weights
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.max_its = max_its
        self.precision = precision

    def fit(self, X, y=None):
        """
        Choosing the best gamma value for the dataset (X,y) using cross-validation.
        Parameters
        ----------
        X: samples
        y: class labels

        Returns nothing
        -------

        """
        n_samples, self.n_features = X.shape
        self.X = X
        self.y = y
        prev_delta = np.Infinity  # previous delta between the old and new values for gamma
        it = 0  # iteration counter
        self.gamma = self.initial_gamma  # reset the value of gamma, learning should start from scratch

        while prev_delta > self.precision and it < self.max_its and self.gamma > 0:
            sum_of_deltas = 0
            for _ in range(self.batch_size):
                rnd_sample_index = np.random.randint(n_samples)
                sum_of_deltas += self.calculate_one_gradient(rnd_sample_index)

            delta = self.learning_rate * sum_of_deltas
            prev_delta = abs(delta)
            self.gamma = self.gamma - delta
            it += 1
            if self.verbose and it % 100 == 0:
                print(f'Iteration {it}: gamma is {self.gamma}.')
            if it % 1000 == 0:
                self.learning_rate /= 10
        if self.gamma <= 0:
            self.gamma += abs(prev_delta)
        if self.verbose:
            print(f'Iteration stopped on iteration {it} with final delta of {prev_delta} and gamma of {self.gamma}')

    def get_metric(self) -> DistanceFunction:
        return KernelDistance(lambda a, b: self.kernel(a, b, self.gamma))

    def calculate_one_gradient(self, index):
        x_0 = self.X[index]
        y_0 = self.y[index]

        # calculate distances and get closest and furthest samples
        closest = np.zeros((self.k, self.n_features))
        closest_distance = np.full(self.k, np.inf)
        furthest = np.zeros((self.k, self.n_features))
        furthest_distance = np.zeros(self.k)
        for i, sample in enumerate(self.X):
            if i != index:
                d = self.kernel(x_0, sample, self.gamma)
                j = 0
                if self.y[i] == y_0:
                    # same class, determine whether sample is one of k closest and its ranking if it is
                    while j < self.k and d > closest_distance[j]:
                        j += 1
                    # shift all further items 1 up
                    new_sample = sample
                    new_distance = d
                    while j < self.k:
                        temp = closest[j]
                        temp_distance = closest_distance[j]
                        closest[j, :] = new_sample
                        closest_distance[j] = new_distance
                        new_sample = temp
                        new_distance = temp_distance
                        j += 1
                else:
                    # different class, determine whether sample is one of k furthest and its ranking if it is
                    while j < self.k and d < furthest_distance[j]:
                        j += 1
                    # shift all further items 1 up
                    new_sample = sample
                    new_distance = d
                    while j < self.k:
                        temp = furthest[j]
                        temp_distance = furthest_distance[j]
                        furthest[j, :] = new_sample
                        furthest_distance[j] = new_distance
                        new_sample = temp
                        new_distance = temp_distance
                        j += 1

        # reverse sort of the furthest elements: furthest element must be assigned the largest weight later
        furthest = np.flip(furthest, 0)
        furthest_distance = np.flip(furthest_distance)

        # calculate the gradient
        weights_0 = self.weights(self.k)
        class_membership_degree = np.dot(weights_0, closest_distance) + \
                                  np.dot(np.flip(weights_0), np.ones(self.k) - furthest_distance)
        s = 0
        for upper_i in range(self.k):
            # upper
            s += weights_0[upper_i] * self.gradient(x_0, closest[upper_i], self.gamma)
            # lower
            lower_i = self.k - upper_i - 1
            s -= weights_0[lower_i] * self.gradient(x_0, furthest[lower_i], self.gamma)
        return -1 * s / class_membership_degree

    @staticmethod
    def get_name():
        return "gen-grad"


class GaussianGradientKernelFactory(GradientKernelFactory):
    def __init__(self,
                 gamma=1,
                 batch_size=10,
                 k=3,
                 weights=LinearWeights(),
                 learning_rate=0.01,
                 max_its=10000,
                 precision=0.00001,
                 verbose=False):
        super().__init__(kernel=lambda a, b, g: np.exp(-1 * (np.linalg.norm(a - b) ** 2) / g),
                         gradient=lambda a, b, g: (np.linalg.norm(a - b) ** 2) / (g ** 2) * np.exp(-1 * (np.linalg.norm(a - b) ** 2) / g),
                         gamma=gamma,
                         batch_size=batch_size,
                         k=k,
                         weights=weights,
                         verbose=verbose,
                         learning_rate=learning_rate,
                         max_its=max_its,
                         precision=precision)

    @staticmethod
    def get_name():
        return "gauss-gradn"


class ExponentialGradientKernelFactory(GradientKernelFactory):
    def __init__(self,
                 gamma=1,
                 batch_size=10,
                 k=3,
                 weights=LinearWeights(),
                 learning_rate=0.01,
                 max_its=10000,
                 precision=0.00001,
                 verbose=False):
        super().__init__(kernel=lambda a, b, g: np.exp(-1 * np.linalg.norm(a - b) / g),
                         gradient=lambda a, b, g: (np.linalg.norm(a - b)) / (g ** 2) * np.exp(-1 * np.linalg.norm(a - b) / g),
                         gamma=gamma,
                         batch_size=batch_size,
                         k=k,
                         weights=weights,
                         verbose=verbose,
                         learning_rate=learning_rate,
                         max_its=max_its,
                         precision=precision)

    @staticmethod
    def get_name():
        return "exp-gradn"


class RationalGradientKernelFactory(GradientKernelFactory):
    def __init__(self,
                 gamma=1,
                 batch_size=10,
                 k=3,
                 weights=LinearWeights(),
                 learning_rate=0.01,
                 max_its=10000,
                 precision=0.00001,
                 verbose=False):
        super().__init__(kernel=lambda a, b, g: g / ((np.linalg.norm(a - b) ** 2) + g),
                         gradient=lambda a, b, g: np.linalg.norm(a - b) ** 2 / ((np.linalg.norm(a - b) ** 2 + g) ** 2), # forgot to square the norm here
                         gamma=gamma,
                         batch_size=batch_size,
                         k=k,
                         weights=weights,
                         verbose=verbose,
                         learning_rate=learning_rate,
                         max_its=max_its,
                         precision=precision)

    @staticmethod
    def get_name():
        return "ratqua-gradn"


def circular_gradient(a, b, g):
    if np.linalg.norm(a - b) < g:
        return 4 / np.pi * np.linalg.norm(a - b) / np.square(g) * np.sqrt(1 - np.square(np.linalg.norm(a - b) / g))
    return 0


class CircularGradientKernelFactory(GradientKernelFactory):
    def __init__(self,
                 gamma=1,
                 batch_size=10,
                 k=3,
                 weights=LinearWeights(),
                 learning_rate=0.01,
                 max_its=10000,
                 precision=0.00001,
                 verbose=False):
        super().__init__(kernel=circular_kernel,
                         gradient=circular_gradient,
                         gamma=gamma,
                         batch_size=batch_size,
                         k=k,
                         weights=weights,
                         verbose=verbose,
                         learning_rate=learning_rate,
                         max_its=max_its,
                         precision=precision)

    @staticmethod
    def get_name():
        return "cir-gradn"


def spherical_gradient(a, b, g):
    if np.linalg.norm(a - b) < g:
        return 3/2 * np.linalg.norm(a-b) / np.square(g) * (1 - np.square(np.linalg.norm(a - b)/g))
    return 0


class SphericalGradientKernelFactory(GradientKernelFactory):
    def __init__(self,
                 gamma=1,
                 batch_size=10,
                 k=3,
                 weights=LinearWeights(),
                 learning_rate=0.01,
                 max_its=10000,
                 precision=0.00001,
                 verbose=False):
        super().__init__(kernel=spherical_kernel,
                         gradient=spherical_gradient,
                         gamma=gamma,
                         batch_size=batch_size,
                         k=k,
                         weights=weights,
                         verbose=verbose,
                         learning_rate=learning_rate,
                         max_its=max_its,
                         precision=precision)

    @staticmethod
    def get_name():
        return "sph-gradn-2"


class ClassMahalanobisDistanceFunction(DistanceFunction):
    """
    Class specific Mahalanobis pseudo-metric
    """

    __slots__ = ('matrix_dict', 'overall_matrix', 'X', 'y',)

    def __init__(self, matrix_dict, overall_matrix, X, y):
        self.matrix_dict = matrix_dict
        self.overall_matrix = overall_matrix
        self.X = X
        self.y = y

    def __call__(self, a, b):
        m = self.overall_matrix
        # look for one of the samples in the training set and select its class
        ind_array = np.where(np.isclose(a, self.X).all(axis=1))[0]
        if len(ind_array) > 0:
            first_class = self.y[ind_array[0]]
            i = 1
            while i < len(ind_array) and self.y[ind_array[i]] == first_class:
                i += 1
            if i == len(ind_array):
                m = self.matrix_dict[self.y[ind_array[0]]]
        else:
            ind_array = np.where(np.isclose(b, self.X).all(axis=1))[0]
            if len(ind_array) > 0:
                first_class = self.y[ind_array[0]]
                i = 1
                while i < len(ind_array) and self.y[ind_array[i]] == first_class:
                    i += 1
                if i == len(ind_array):
                    m = self.matrix_dict[self.y[ind_array[0]]]
        return np.sqrt(np.matmul(np.matmul(np.transpose(a - b), m), (a - b)))


class ClassMahalanobisDistanceFactory(DistanceFunctionFactory):
    """
    Factory that creates fitted, class specific Mahalanobis distances.
    """

    __slots__ = ('matrix_dict', 'general_matrix', 'X', 'y', )

    def fit(self, X, y=None):
        if self.can_apply(X):
            self.general_matrix = np.linalg.inv(np.cov(X, rowvar=False))
            # calculate the inverse of the covariance matrix of the set of elements of each class
            classes = np.unique(y)
            self.matrix_dict = dict()
            for c in classes:
                self.matrix_dict[c] = self.general_matrix
                c_index = np.where(y == c)[0]
                # if there is only one element of this class, the covariance matrix is zero
                if len(c_index) > 1:
                    c_x = X[c_index]
                    c_matrix = np.linalg.inv(np.cov(c_x, rowvar=False))
                    self.matrix_dict[c] = c_matrix


            self.X = X
            self.y = y

    def get_metric(self) -> DistanceFunction:
        return ClassMahalanobisDistanceFunction(self.matrix_dict, self.general_matrix, self.X, self.y)

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


class ClassMaxMahalanobisDistanceFunction(DistanceFunction):
    """
    Class specific Mahalanobis pseudo-metric
    """

    __slots__ = ('matrix_dict', 'overall_matrix', 'overal_max_d', 'max_d_dict', 'X', 'y',)

    def __init__(self, matrix_dict, overall_matrix, max_d_dict, overal_max_d, X, y):
        self.matrix_dict = matrix_dict
        self.overall_matrix = overall_matrix
        self.max_d_dict = max_d_dict
        self.overal_max_d = overal_max_d
        self.X = X
        self.y = y

    def __call__(self, a, b):
        m = self.overall_matrix
        d = self.overal_max_d
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
        return np.sqrt(np.matmul(np.matmul(np.transpose(a - b), m), (a - b))) / d


class ClassMaxMahalanobisDistanceFactory(DistanceFunctionFactory):
    """
    Factory that creates fitted, class specific Mahalanobis distances.
    """

    __slots__ = ('matrix_dict', 'general_matrix', 'overal_max_d', 'max_d_dict', 'X', 'y',)

    def fit(self, X, y=None):
        if self.can_apply(X):
            self.general_matrix = np.linalg.inv(np.cov(X, rowvar=False))
            self.overal_max_d = 0
            for x1 in X:
                for x2 in X:
                    d12 = np.sqrt(np.matmul(np.matmul(np.transpose(x1 - x2), self.general_matrix), (x1 - x2)))
                    if d12 > self.overal_max_d:
                        self.overal_max_d = d12
            # calculate the inverse of the covariance matrix of the set of elements of each class
            classes = np.unique(y)
            self.matrix_dict = dict()
            self.max_d_dict = dict()
            for c in classes:
                self.matrix_dict[c] = self.general_matrix
                c_index = np.where(y == c)[0]
                # if there is only one element of this class, the covariance matrix is zero
                if len(c_index) > 1:
                    c_x = X[c_index]
                    c_matrix = np.linalg.inv(np.cov(c_x, rowvar=False))
                    self.matrix_dict[c] = c_matrix
                    d = 0
                    for x1 in X:
                        for x2 in X:
                            d12 = np.sqrt(np.matmul(np.matmul(np.transpose(x1 - x2), self.general_matrix), (x1 - x2)))
                            if d12 > d:
                                d = d12
                    self.max_d_dict[c] = d
            self.X = X
            self.y = y

    def get_metric(self) -> DistanceFunction:
        return ClassMaxMahalanobisDistanceFunction(self.matrix_dict,
                                                   self.general_matrix,
                                                   self.max_d_dict,
                                                   self.overal_max_d,
                                                   self.X,
                                                   self.y)

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
        return "cl-max-mah"


# i did not even use the c-matrix when calculating the max distance per class woooow
class DefaultClassMaxMahalanobisDistanceFactory(DistanceFunctionFactory):
    """
    Factory that creates fitted, class specific Mahalanobis distances.
    """

    __slots__ = ('matrix_dict', 'general_matrix', 'overal_max_d', 'max_d_dict', 'X', 'y',)

    def fit(self, X, y=None):
        if self.can_apply(X):
            self.general_matrix = np.linalg.inv(np.cov(X, rowvar=False))
            self.overal_max_d = 0
            for x1 in X:
                for x2 in X:
                    d12 = np.sqrt(np.matmul(np.matmul(np.transpose(x1 - x2), self.general_matrix), (x1 - x2)))
                    if d12 > self.overal_max_d:
                        self.overal_max_d = d12
            # calculate the inverse of the covariance matrix of the set of elements of each class
            classes = np.unique(y)
            self.matrix_dict = dict()
            self.max_d_dict = dict()
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
                    d = 0
                    for x1 in c_x:
                        for x2 in X:
                            d12 = np.sqrt(np.matmul(np.matmul(np.transpose(x1 - x2), c_matrix), (x1 - x2)))
                            if d12 > d:
                                d = d12
                    self.max_d_dict[c] = d
            self.X = X
            self.y = y

    def get_metric(self) -> DistanceFunction:
        return ClassMaxMahalanobisDistanceFunction(self.matrix_dict,
                                                   self.general_matrix,
                                                   self.max_d_dict,
                                                   self.overal_max_d,
                                                   self.X,
                                                   self.y)

    def can_apply(self, X, y=None) -> bool:
        cov = np.cov(X, rowvar=False)
        return np.linalg.matrix_rank(cov) == cov.shape[0]

    @staticmethod
    def get_name():
        return "cl-max-mah-def"


class ClassKernelDistanceFunction(DistanceFunction):
    """
    Class specific kernel based distance
    ---

    Kernel needs to be a function with 3 arguments: a, b, gamma
    """

    __slots__ = ('kernel', 'gamma_dict', 'overall_gamma', 'X', 'y',)

    def __init__(self, kernel, gamma_dict, overall_gamma, X, y):
        self.kernel = kernel
        self.gamma_dict = gamma_dict
        self.overall_gamma = overall_gamma
        self.X = X
        self.y = y

    def __call__(self, a, b):
        g = self.overall_gamma
        # look for one of the samples in the training set and select its class
        ind_array = np.where(np.isclose(a, self.X).all(axis=1))[0]
        if len(ind_array) > 0:
            first_class = self.y[ind_array[0]]
            i = 1
            while i < len(ind_array) and self.y[ind_array[i]] == first_class:
                i += 1
            if i == len(ind_array):
                g = self.gamma_dict[first_class]
        else:
            ind_array = np.where(np.isclose(b, self.X).all(axis=1))[0]
            if len(ind_array) > 0:
                first_class = self.y[ind_array[0]]
                i = 1
                while i < len(ind_array) and self.y[ind_array[i]] == first_class:
                    i += 1
                if i == len(ind_array):
                    g = self.gamma_dict[first_class]
        return self.kernel(a, b, g)


# todo
class ClassKernelDistanceFactory(DistanceFunctionFactory, ABC):
    """
    Factory that creates fitted, class specific distances based on kernelised relations.
    """

    __slots__ = ('kernel', 'gamma_dict', 'overall_gamma', 'X', 'y', 'gradient', 'batch_size', 'k', 'weights',
                 'learning_rate', 'max_its', 'precision', 'verbose')

    def __init__(self,
                 kernel,
                 gradient,
                 gamma=1,
                 batch_size=10,
                 k=3,
                 weights=LinearWeights(),
                 learning_rate=0.01,
                 max_its=10000,
                 precision=0.00001,
                 verbose=False):
        """

        Parameters
        ----------
        kernel          lambda (a,b, gamma) -> kernel_gamma(a,b)
        gradient        lambda (a,b, gamma) -> gradient_gamma(a,b)
        gamma           initial value for gamma
        batch_size      number of gradients to calculate in each step
        k               number of neighbours used in FRNN
        weights         used in FRNN
        learning_rate   represents the size of the steps taken in gradient descent
        max_its         maximum number of iterations
        precision       minimum difference in gamma between two subsequent iterations
        verbose         print stuff or not
        """
        self.kernel = kernel
        self.gradient = gradient
        self.overall_gamma = gamma
        self.batch_size = batch_size
        self.k = k
        self.weights = weights
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.max_its = max_its
        self.precision = precision

    def fit(self, X, y=None):
            self.X = X
            self.y = y
            # we will use gradient descent to learn a different gamma for each class
            classes = np.unique(y)
            self.gamma_dict = dict()
            for c in classes:
                g = self.overall_gamma
                c_index = np.where(y == c)[0]
                c_x = self.X[c_index]
                n_samples, n_features = c_x.shape
                prev_delta = np.Infinity  # previous delta between the old and new values for gamma
                it = 0  # iteration counter

                while prev_delta > self.precision and it < self.max_its:
                    sum_of_deltas = 0
                    for _ in range(self.batch_size):
                        rnd_sample_index = np.random.randint(n_samples)
                        sum_of_deltas += self.calculate_one_gradient(rnd_sample_index, n_features, g)

                    delta = self.learning_rate * sum_of_deltas
                    prev_delta = abs(delta)
                    g = g - delta
                    it += 1
                    if self.verbose and it % 10 == 0:
                        print(f'Class {c}, iteration {it}: gamma is {self.overall_gamma}.')
                    if it % 1000 == 0:
                        self.learning_rate /= 10
                self.gamma_dict[c] = g

    def calculate_one_gradient(self, index, n_features, g):
        x_0 = self.X[index]
        y_0 = self.y[index]

        # calculate distances and get closest and furthest samples
        closest = np.zeros((self.k, n_features))
        closest_distance = np.full(self.k, np.inf)
        furthest = np.zeros((self.k, n_features))
        furthest_distance = np.zeros(self.k)
        for i, sample in enumerate(self.X):
            if i != index:
                d = self.kernel(x_0, sample, g)
                j = 0
                if self.y[i] == y_0:
                    # same class, determine whether sample is one of k closest and its ranking if it is
                    while j < self.k and d > closest_distance[j]:
                        j += 1
                    # shift all further items 1 up
                    new_sample = sample
                    new_distance = d
                    while j < self.k:
                        temp = closest[j]
                        temp_distance = closest_distance[j]
                        closest[j, :] = new_sample
                        closest_distance[j] = new_distance
                        new_sample = temp
                        new_distance = temp_distance
                        j += 1
                else:
                    # different class, determine whether sample is one of k furthest and its ranking if it is
                    while j < self.k and d < furthest_distance[j]:
                        j += 1
                    # shift all further items 1 up
                    new_sample = sample
                    new_distance = d
                    while j < self.k:
                        temp = furthest[j]
                        temp_distance = furthest_distance[j]
                        furthest[j, :] = new_sample
                        furthest_distance[j] = new_distance
                        new_sample = temp
                        new_distance = temp_distance
                        j += 1

        # reverse sort of the furthest elements: furthest element must be assigned the largest weight later
        furthest = np.flip(furthest, 0)
        furthest_distance = np.flip(furthest_distance)

        # calculate the gradient
        weights_0 = self.weights(self.k)
        class_membership_degree = np.dot(weights_0, closest_distance) + \
                                  np.dot(np.flip(weights_0), np.ones(self.k) - furthest_distance)
        s = 0
        for upper_i in range(self.k):
            # upper
            s += weights_0[upper_i] * self.gradient(x_0, closest[upper_i], g)
            # lower
            lower_i = self.k - upper_i - 1
            s -= weights_0[lower_i] * self.gradient(x_0, furthest[lower_i], g)
        return -1 * s / class_membership_degree

    def get_metric(self) -> DistanceFunction:
        return ClassKernelDistanceFunction(self.kernel, self.gamma_dict, self.overall_gamma, self.X, self.y)

    @staticmethod
    def get_name():
        return "cl-ker-gen"


class ClassGaussianKernelFactory(ClassKernelDistanceFactory):
    def __init__(self,
                 gamma=1,
                 batch_size=10,
                 k=3,
                 weights=LinearWeights(),
                 learning_rate=0.01,
                 max_its=10000,
                 precision=0.00001,
                 verbose=False):
        super().__init__(kernel=lambda a, b, g: np.exp(-1 * (np.linalg.norm(a - b) ** 2) / g),
                         gradient=lambda a, b, g: (np.linalg.norm(a - b) ** 2) / (g ** 2) * np.exp(-1 * (np.linalg.norm(a - b) ** 2) / g),
                         gamma=gamma,
                         batch_size=batch_size,
                         k=k,
                         weights=weights,
                         verbose=verbose,
                         learning_rate=learning_rate,
                         max_its=max_its,
                         precision=precision)

    @staticmethod
    def get_name():
        return "gauss-class"


class ClassNCAFactory(DistanceFunctionFactory):
    """
    Factory that creates fitted, class specific NCA based distance, as in the MaxClassDefault mahalanobis
    distance, but with NCA instead of the covariance matrix.
    It uses the identity matrix as a fallback.
    """

    def __init__(self, k=3):
        self.k = k
        self.model = ClassNCA()

    def fit(self, X, y=None):
        """
        In the fit function, we hand of the training of CLNCA to the ClassNCA class, but we still have to learn
        the maximum distance, such that we can make sure that we end up having fuzzy relations with legal membership
        functions.

        Parameters
        ----------
        X
        y

        Returns
        -------

        """
        self.model.fit(X=X, y=y)
        self.matrix_dict = self.model.metric()

        # calculate overall max distance
        self.overal_max_d = 0
        for x1 in X:
            for x2 in X:
                d12 = np.sqrt(np.matmul(np.transpose(x1 - x2), (x1 - x2)))
                if d12 > self.overal_max_d:
                    self.overal_max_d = d12
        if self.overal_max_d == 0:
            self.overal_max_d = 1

        # calculate max distance on each class
        self.max_d_dict = dict()
        for c in np.unique(y):
            d = 0
            for x1 in X[np.where(y == c)[0]]:
                for x2 in X:
                    d12 = np.sqrt(np.matmul(np.matmul(np.transpose(x1 - x2), self.matrix_dict[c]), (x1 - x2)))
                    if d12 > d:
                        d = d12
            if d == 0:
                d = 1
            self.max_d_dict[c] = d
        self.X = X
        self.y = y

    def get_metric(self) -> DistanceFunction:
        return ClassMaxMahalanobisDistanceFunction(self.matrix_dict,
                                                   np.identity(self.X.shape[1]),
                                                   self.max_d_dict,
                                                   self.overal_max_d,
                                                   self.X,
                                                   self.y)

    def can_apply(self, X, y=None) -> bool:
        return X.shape[0] < 4000 and neighbour_applier(y, self.k)

    @staticmethod
    def get_name():
        return "cl-NCA"


class SquaredClassNCAFactory(DistanceFunctionFactory):
    """
    Factory that creates fitted, class specific NCA based distance, as in the MaxClassDefault mahalanobis
    distance, but with NCA instead of the covariance matrix.
    It uses the identity matrix as a fallback.
    Moreover, the resulting metric uses the squared Euclidean distance in the transformed space.
    """

    def __init__(self, k=3):
        self.k = k
        self.model = ClassNCA()

    def fit(self, X, y=None):
        """
        In the fit function, we hand of the training of CLNCA to the ClassNCA class, but we still have to learn
        the maximum distance, such that we can make sure that we end up having fuzzy relations with legal membership
        functions.

        Parameters
        ----------
        X training samples
        y class labels

        Returns nothing
        """
        self.model.fit(X=X, y=y)
        self.matrix_dict = self.model.metric()

        # calculate overall max distance
        self.overal_max_d = 0
        for x1 in X:
            for x2 in X:
                d12 = np.sqrt(np.matmul(np.transpose(x1 - x2), (x1 - x2)))
                if d12 > self.overal_max_d:
                    self.overal_max_d = d12
        if self.overal_max_d == 0:
            self.overal_max_d = 1

        # calculate max distance on each class
        self.max_d_dict = dict()
        for c in np.unique(y):
            d = 0
            for x1 in X[np.where(y == c)[0]]:
                for x2 in X:
                    d12 = np.sqrt(np.matmul(np.matmul(np.transpose(x1 - x2), self.matrix_dict[c]), (x1 - x2)))
                    if d12 > d:
                        d = d12
            if d == 0:
                d = 1
            self.max_d_dict[c] = d
        self.X = X
        self.y = y

    def get_metric(self) -> DistanceFunction:
        return ClassMaxMahalanobisDistanceFunction(self.matrix_dict,
                                                   np.identity(self.X.shape[1]),
                                                   self.max_d_dict,
                                                   self.overal_max_d,
                                                   self.X,
                                                   self.y)

    def can_apply(self, X, y=None) -> bool:
        return X.shape[0] < 4000 and neighbour_applier(y, self.k)

    @staticmethod
    def get_name():
        return "clNCAsq"

