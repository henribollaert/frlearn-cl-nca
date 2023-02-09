import numpy as np
from abc import ABC
from frlearn.uncategorised.weights import LinearWeights
from .relations_base import DistanceFunction, DistanceFunctionFactory


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


class KernelFactory(DistanceFunctionFactory, ABC):
    """
    Factory for distance based on a relation based on the Gaussian kernel.
    """
    _slots__ = ('kernel', 'gamma', '_gamma', )

    def __init__(self, kernel, gamma="auto"):
        """
        Initialisation of the kernel factory.
        Parameters
        ----------
        kernel  lambda that takes a gamma parameter and returns the kernel function with that parmater, which
                is also a lambda funciton with 2 parameters
        gamma   either the string "auto", in which case gamma will be selected as 1 / nr_of_features of the data set,
                or a numeric value (int or float)
        """
        self.kernel = kernel
        self.gamma = gamma
        self._gamma = 1

    def fit(self, X, y=None):
        if self.gamma == "auto":
            self._gamma = X.shape[1]
        elif type(self.gamma) == int or type(self.gamma) == float:
            self._gamma = self.gamma

    def get_metric(self) -> DistanceFunction:
        return KernelDistance(self.kernel(self._gamma))

    @staticmethod
    def get_name():
        return "generic-kernel"


class GaussianKernelFactory(KernelFactory):
    """
    Factory for distance based on a relation based on the Gaussian kernel.
    """

    __slots__ = ('gamma', 'folds', 'k', 'weights', 'verbose')

    def __init__(self, gamma="auto"):
        super(GaussianKernelFactory, self).__init__(
            kernel=lambda g: (lambda a, b: np.exp(-1 * np.linalg.norm(a - b) ** 2 / g)),
            gamma=gamma
        )

    @staticmethod
    def get_name():
        return "gauss-mult"


class ExponentialKernelFactory(KernelFactory):
    """
    Factory for distance based on a relation based on the exponential kernel.
    """

    def __init__(self, gamma="auto"):
        super(ExponentialKernelFactory, self).__init__(
            kernel=lambda g: (lambda a, b: np.exp(-1 * np.linalg.norm(a - b) / g)),
            gamma=gamma
        )

    @staticmethod
    def get_name():
        return "exp-mult"


class RationalQuadraticKernelFactory(KernelFactory):
    """
    Factory for distance based on a relation based on the rational quadratic kernel.
    """

    def __init__(self, gamma="auto"):
        super(RationalQuadraticKernelFactory, self).__init__(
            kernel=lambda g: (lambda a, b: g / (np.square(np.linalg.norm(a - b)) + g)),
            gamma=gamma
        )

    @staticmethod
    def get_name():
        return "rat-mult"


def circular_kernel(a, b, g):
    if np.linalg.norm(a - b) < g:
        t1 = 2 / np.pi * np.arccos(np.linalg.norm(a - b) / g)
        t2 = 2 / np.pi * np.linalg.norm(a - b) / g * np.sqrt(1 - np.square(np.linalg.norm(a - b) / g))
        return t1 - t2
    return 0


class CircularKernelFactory(KernelFactory):
    """
    Factory for distance based on a relation based on the Gaussian kernel.
    """

    def __init__(self, gamma="auto"):
        super(CircularKernelFactory, self).__init__(
            kernel=lambda g: (lambda a, b: circular_kernel(a, b, g)),
            gamma=gamma
        )

    @staticmethod
    def get_name():
        return "circle-mult"


def spherical_kernel(a, b, g):
    if np.linalg.norm(a - b) < g:
        t1 = 3 / 2 * np.linalg.norm(a - b) / g
        t2 = 1 / 2 * np.power(np.linalg.norm(a - b) / g, 3)
        return 1 - t1 + t2
    return 0


class SphericalKernelFactory(KernelFactory):
    """
    Factory for distance based on a relation based on the Gaussian kernel.
    """

    def __init__(self, gamma="auto"):
        super(SphericalKernelFactory, self).__init__(
            kernel=lambda g: (lambda a, b: spherical_kernel(a, b, g)),
            gamma=gamma
        )

    @staticmethod
    def get_name():
        return "sphere-mult"


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
        weights         used in FRNN
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
        class_membership_degree = \
            np.dot(weights_0, closest_distance) + np.dot(np.flip(weights_0), np.ones(self.k) - furthest_distance)
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
                         gradient=lambda a, b, g: (np.linalg.norm(a - b) ** 2) / (g ** 2) * np.exp(
                             -1 * (np.linalg.norm(a - b) ** 2) / g),
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
        return "gauss-grad"


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
                         gradient=lambda a, b, g: (np.linalg.norm(a - b)) / (g ** 2) * np.exp(
                             -1 * np.linalg.norm(a - b) / g),
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
        return "exp-grad"


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
                         gradient=lambda a, b, g: np.linalg.norm(a - b) ** 2 / ((np.linalg.norm(a - b) ** 2 + g) ** 2),
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
        return "rat-grad"


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
        return "cir-grad"


def spherical_gradient(a, b, g):
    if np.linalg.norm(a - b) < g:
        return 3 / 2 * np.linalg.norm(a - b) / np.square(g) * (1 - np.square(np.linalg.norm(a - b) / g))
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
        return "sph-grad"
