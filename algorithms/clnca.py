"""
CLass-specific
Neighbourhood Component Analysis (NCA)

"""

from __future__ import absolute_import
import numpy as np
from six.moves import xrange
from sklearn.utils.validation import check_X_y
from sklearn.metrics.pairwise import euclidean_distances

from .dml_utils import calc_outers, calc_outers_i
from .dml_algorithm import DML_Algorithm


class ClassNCA(DML_Algorithm):
    """
    Class-specific Neighborhood Component Analysis (CNCA)

    A distance metric learning algorithm that tries to minimize kNN expected error.

    Parameters
    ----------

    learning_rate : string, default='adaptive'

        Type of learning rate update for gradient descent. Possible values are:

        - 'adaptive' : the learning rate will increase if the gradient step is successful, else it will decrease.

        - 'constant' : the learning rate will be constant during all the gradient steps.

    eta0 : int, default=0.3

        The initial value for learning rate.

    initial_transform : 2D-Array or Matrix (d' x d), or string, default=None.

        If array or matrix that will represent the starting linear map for gradient descent, where d is the number of features,
        and d' is the dimension specified in num_dims.
        If None, euclidean distance will be used. If a string, the following values are allowed:

        - 'euclidean' : the euclidean distance.

        - 'scale' : a diagonal matrix that normalizes each attribute according to its range will be used.

    max_iter : int, default=100

        Maximum number of gradient descent iterations.

    prec : float, default=1e-8

        Precision stop criterion (gradient norm).

    tol : float, default=1e-8

        Tolerance stop criterion (difference between two iterations)

    eta_thres : float, default=1e-14

        A learning rate threshold stop criterion.

    learn_inc : float, default=1.01

        Increase factor for learning rate. Ignored if learning_rate is not 'adaptive'.

    learn_dec : float, default=0.5

        Decrease factor for learning rate. Ignored if learning_rate is not 'adaptive'.

    References
    ----------
        Jacob Goldberger et al. “Neighbourhood components analysis”. In: Advances in neural
        information processing systems. 2005, pages 513-520.

    """

    def __init__(self,
                 learning_rate="adaptive",
                 eta0=0.3,
                 max_iter=1000,
                 prec=1e-8,
                 tol=1e-8,
                 eta_thres=1e-14,
                 learn_inc=1.01,
                 learn_dec=0.5):
        self.max_iter = max_iter
        self.eta = self.eta0 = eta0
        self.learning_rate = learning_rate
        self.adaptive_ = (self.learning_rate == 'adaptive')
        self.prec = prec
        self.tol = tol
        self.eta_thres = eta_thres
        self.learn_inc = learn_inc
        self.learn_dec = learn_dec

        # Metadata initialization
        self.num_its_ = None
        self.initial_softmax_ = None
        self.final_softmax_ = None
        
        # super(NCA, self).__init__()

    def metadata(self):
        """
        Obtains algorithm metadata.

        Returns
        -------
        meta : A dictionary with the following metadata:
            - num_iters : Number of iterations that the descent method took.

            - initial_expectance : Initial value of the objective function (the expected LOO score)

            - final_expectance : Final value of the objective function (the expected LOO score)
        """
        return {'num_iters': self.num_its_, 'initial_expectance': self.initial_softmax_, 'final_expectance': self.final_softmax_}

    def transformer(self):
        """
        Obtains the learned projection.

        Returns
        -------
        General_matrix : (dxd) matrix, where  d is the number of features.
        and matrix dict -> transformations!
        """
        return self.general_matrix, self.matrix_dict

    def metric(self):
        """
        Computes the Mahalanobis matrices from the transformation matrices.

        Returns
        -------

        """
        mah_dict = dict()
        for c, matrix in self.matrix_dict.items():
            mah_dict[c] = matrix.T @ matrix
        return mah_dict

    def fit(self, X, y):
        """
        Fit the model from the data in X and the labels in y.

        Parameters
        ----------
        X : array-like, shape (N x d)
            Training vector, where N is the number of samples, and d is the number of features.

        y : array-like, shape (N)
            Labels vector, where N is the number of samples.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.n_, self.d_ = X.shape

        self.eta = self.eta0
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        # initialise general_matrix and class specific matrices
        L = np.zeros([self.d_, self.d_])
        np.fill_diagonal(L, 1.0)  # Euclidean distance

        general_matrix = L.copy()

        # Get class proportions and indexes
        classes = np.unique(y)
        class_split_inds = {}

        matrix_dict = dict()

        for cc in classes:
            class_split_inds[cc] = np.where(y == cc)[0]
            matrix_dict[cc] = L.copy()

        self.initial_softmax_ = self._compute_expected_success(matrix_dict, X, y, class_split_inds) / len(y)

        self._SGD_fit(X, y, general_matrix, matrix_dict, class_split_inds)

        self.final_softmax_ = self._compute_expected_success(self.matrix_dict, X, y, class_split_inds) / len(y)
        return self

    def _SGD_fit(self, X, y, general_matrix, matrix_dict, class_split_inds):
        # Initialize parameters
        n, d = X.shape
        outers = np.empty([n, n, d, d], dtype=float)  # !!!!

        # for i in xrange(n):
        #     for j in xrange(m):
        #         outers[i, j] = np.outer(X[i, :] - Y[j, :], X[i, :] - Y[j, :])

        n = self.n_
        d = self.d_


        num_its = 0
        max_it = self.max_iter

        grad = None

        succ_prev = 0.0
        succ = 0.0
        eta = self.eta
        etamin = self.eta_thres
        l_inc = self.learn_inc
        l_dec = self.learn_dec
        eps = self.prec
        tol = self.tol

        stop = False
        adaptive = self.adaptive_

        while not stop:
            rnd = np.random.permutation(len(y))

            for i in rnd:
                yi = y[i]  # we will change the matrix corresponding to this class

                # First we need to transform X with the current matrices.
                transformed_space = ClassNCA.transformX(matrix_dict, X, y)

                # Then we need to calculate p_ij (softmax)

                # To avoid exponential underflow we use the identity softmax(x) = softmax(x + c) for all c,
                # and take c = max(dists)
                transformed_sample = transformed_space[i]
                difference_space = transformed_sample - transformed_space
                dists_i = -np.diag(difference_space.dot(difference_space.T))
                dists_i[i] = -np.inf

                i_max = np.argmax(dists_i)
                c = dists_i[i_max]

                softmax = np.empty([n], dtype=float)

                for j in xrange(n):
                    if j != i:
                        # To avoid precision errors, argmax is assigned directly softmax 1
                        if j == i_max:
                            softmax[j] = 1
                        else:
                            pw = min(0, dists_i[j] - c)
                            softmax[j] = np.exp(pw)

                softmax[i] = 0
                softmax /= softmax.sum()

                # This allows us to calculate p_i
                p_i = softmax[class_split_inds[y[i]]].sum()

                # We can now calculate the gradients
                sum_all = np.zeros([d, d])
                sum_same = np.zeros([d, d])

                for k in xrange(n):
                    s = softmax[k] * matrix_dict[y[k]] @ np.outer(X[k, :], X[k, :])
                    sum_all += s
                    if y[i] == y[k]:
                        sum_same += s

                # Now we need to calculate the gradient
                grad = 2 * (p_i * (matrix_dict[yi] @ np.outer(X[i, :], X[i, :]) - sum_all) - sum_same)
                # ... for the class specific matrix
                matrix_dict[yi] += eta * grad

            # calculate objective function
            # not divided by len(y) here, why? doesn't matter not compared to that
            succ = self._compute_expected_success(matrix_dict, X, y, class_split_inds)

            # technical details for SGD loop
            if adaptive:
                if succ > succ_prev:
                    eta *= l_inc
                else:
                    eta *= l_dec
                    if eta < etamin:
                        stop = True

                succ_prev = succ

            grad_norm = np.max(np.abs(grad))
            if grad_norm < eps or eta * grad_norm < tol:  # Difference between two iterations is given by eta*grad
                stop = True

            # print(num_its)
            num_its += 1
            if num_its == max_it:
                stop = True
            if stop:
                break

        self.num_its_ = num_its
        self.eta = eta
        self.general_matrix = general_matrix
        self.matrix_dict = matrix_dict

        return self

    @staticmethod
    def _compute_expected_success(matrix_dict, X, y, class_split_inds=None):
        """
        Computes the sum over all i and j of p_ij.
        Parameters
        ----------
        matrix_dict
        X
        y
        class_split_inds

        Returns
        -------

        """
        n, d = X.shape
        transformed_space = ClassNCA.transformX(matrix_dict, X, y)
        success = 0.0

        dists = euclidean_distances(transformed_space)
        for i in range(len(y)):
            softmax = np.empty([n], dtype=float)
            dists_i = -dists[i, :]
            dists_i[i] = -np.inf
            i_max = np.argmax(dists_i)
            c = dists_i[i_max]
            for j in xrange(n):
                if j != i:
                    if j == i_max:
                        softmax[j] = 1
                    else:
                        pw = min(0, dists_i[j] - c)
                        softmax[j] = np.exp(pw)
            softmax[i] = 0
            softmax /= softmax.sum()

            # Calc p_i
            yi_mask = np.where(y == y[i])[0] if class_split_inds is None else class_split_inds[y[i]]
            p_i = softmax[yi_mask].sum()

            success += p_i

        return success

    # works, this is what psi(x) must be
    @staticmethod
    def transformX(matrix_dict, X, y):
        transformed_x = np.zeros(X.shape)
        for i in range(len(y)):
            transformed_x[i, :] = matrix_dict[y[i]].dot(X[i, :].T).T
        return transformed_x
