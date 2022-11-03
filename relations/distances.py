import numpy as np
from frlearn.vector_size_measures import MinkowskiSize
from frlearn.uncategorised.weights import LinearWeights
from frlearn.neighbour_search_methods import BallTree
from frlearn.classifiers import FRNN
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score
from frlearn.base import select_class

from relations_base import DistanceFunction, DistanceFunctionFactory
from mahalanobis import MahalanobisCorrelationDistanceFactory


def get_new_factories():
    factories = [ManhattanDistanceFactory(),
                 EuclideanDistanceFactory(),
                 ChebyshevDistanceFactory(),
                 CosineMeasureFactory(),
                 CanberraFactory(),
                 MahalanobisCorrelationDistanceFactory()]
    return factories


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
        return "che"

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
        return CorrelationFactory.CorrelationDistance(self.averages)

    @staticmethod
    def get_name():
        return "cor"

    class CorrelationDistance(DistanceFunction):
        """
        Correlation distance based on fraction conversion.
        """

        __slots__ = ('averages',)

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

    def get_metric(self) -> DistanceFunction:
        return CosineMeasureFactory.CosineDistance()

    def can_apply(self, X, y=None) -> bool:
        return not (X == 0).all(axis=1).any()

    @staticmethod
    def get_name():
        return "cos"

    class CosineDistance(DistanceFunction):
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
        return "combo"
