from abc import ABC, abstractmethod


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
