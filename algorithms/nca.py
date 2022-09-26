"""
Neighbourhood Component Analysis (NCA)

"""

from __future__ import absolute_import
import numpy as np
from six.moves import xrange
from sklearn.utils.validation import check_X_y
from sklearn.metrics.pairwise import euclidean_distances

from .dml_utils import calc_outers, calc_outers_i
from .dml_algorithm import DML_Algorithm


class NCA(DML_Algorithm):
    """
    Neighborhood Component Analysis (NCA)

    A distance metric learning algorithm that tries to minimize kNN expected error.

    Parameters
    ----------

    num_dims : int, default=None

        Desired value for dimensionality reduction. If None, the dimension of transformed data will be the same as the original.

    learning_rate : string, default='adaptive'

        Type of learning rate update for gradient descent. Possible values are:

        - 'adaptive' : the learning rate will increase if the gradient step is succesful, else it will decrease.

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

    descent_method : string, default='SGD'

        The descent method to use. Allowed values are:

        - 'SGD' : stochastic gradient descent.

        - 'BGD' : batch gradient descent.

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
                 num_dims=None,
                 learning_rate="adaptive",
                 eta0=0.3,
                 initial_transform=None,
                 max_iter=100,
                 prec=1e-8,
                 tol=1e-8,
                 descent_method="SGD",
                 eta_thres=1e-14,
                 learn_inc=1.01,
                 learn_dec=0.5):
        self.num_dims = num_dims
        self.initial_transform = initial_transform
        self.max_iter = max_iter
        self.eta = self.eta0 = eta0
        self.learning_rate = learning_rate
        self.adaptive_ = (self.learning_rate == 'adaptive')
        self.descent_method = descent_method
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
        L : (d'xd) matrix, where d' is the desired output dimension and d is the number of features.
        """
        return self.L_

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
        if self.num_dims is not None:
            self.nd_ = min(self.d_, self.num_dims)
        else:
            self.nd_ = self.d_

        self.L_ = self.initial_transform
        self.eta = self.eta0
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        if self.L_ is None or self.L_ == "euclidean":
            self.L_ = np.zeros([self.nd_, self.d_])
            np.fill_diagonal(self.L_, 1.0)  # Euclidean distance
        elif self.L_ == "scale":
            self.L_ = np.zeros([self.nd_, self.d_])
            np.fill_diagonal(self.L_, 1. / (np.maximum(X.max(axis=0) - X.min(axis=0), 1e-16)))  # Scaled eculidean distance

        self.initial_softmax_ = self._compute_expected_success(self.L_, X, y) / len(y)

        if self.descent_method == "SGD":  # Stochastic Gradient Descent
            self._SGD_fit(X, y)
        elif self.descent_method == "BGD":  # Batch Gradient Descent
            self._BGD_fit(X, y)

        self.final_softmax_ = self._compute_expected_success(self.L_, X, y) / len(y)
        return self

    def _SGD_fit(self, X, y):
        # Initialize parameters
        outers = calc_outers(X)

        n = self.n_
        d = self.d_

        L = self.L_

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

        # i, j, k, i_max

        # Lx,  sum_p, sum_m, s, Ldiff
        # Lxi, softmax, dists_i, Lxij
        # rnd, yi_mask

        # cdef float c, pw, p_i, grad_norm

        # Get class proportions and indexes
        classes = np.unique(y)
        class_split_inds = {}

        for cc in classes:
            class_split_inds[cc] = np.where(y == cc)[0]

        while not stop:
            # X, y, outers = self._shuffle(X,y,outers)
            rnd = np.random.permutation(len(y))

            for i in rnd:
                # grad = np.zeros([d, d])

                # tt = time.time()
                Lx = L.dot(X.T).T

                # Calc p_ij (softmax)

                # To avoid exponential underflow we use the identity softmax(x) = softmax(x + c) for all c, and take c = max(dists)
                Lxi = Lx[i]
                Ldiff = Lxi - Lx
                dists_i = -np.diag(Ldiff.dot(Ldiff.T))
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

                # Calc p_i
                # yi_mask = np.where(y == y[i])[0]
                p_i = softmax[class_split_inds[y[i]]].sum()

                # Gradient computing
                sum_p = np.zeros([d, d])
                sum_m = np.zeros([d, d])

                outers_i = calc_outers_i(X, outers, i)

                for k in xrange(n):
                    s = softmax[k] * outers_i[k]
                    sum_p += s
                    if(y[i] == y[k]):
                        sum_m -= s

                grad = p_i * sum_p + sum_m
                grad = 2 * L.dot(grad)
                L += eta * grad

            succ = self._compute_expected_success(L, X, y, class_split_inds)
            # print(succ / len(y))

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

            num_its += 1
            if num_its == max_it:
                stop = True
            if stop:
                break

        self.num_its_ = num_its
        self.eta = eta
        self.L_ = L

        return self

    def _BGD_fit(self, X, y):
        # Initialize parameters
        outers = calc_outers(X)

        n, d = self.n_, self.d_

        L = self.L_

        self.num_its_ = 0

        grad = None

        succ_prev = succ = 0.0

        stop = False

        while not stop:
            grad = np.zeros([d, d])
            Lx = L.dot(X.T).T

            succ = 0.0  # Expected error can be computed directly in BGD

            for i, yi in enumerate(y):
                # Calc p_ij (softmax)
                # To avoid exponential underflow we use the identity softmax(x) = softmax(x + c) for all c, and take c = max(dists)
                Lxi = Lx[i]
                dists_i = -np.diag((Lxi - Lx).dot((Lxi - Lx).T))
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
                            pw = min(0, -((Lx[i] - Lx[j]).dot(Lx[i] - Lx[j])) - c)
                            softmax[j] = np.exp(pw)

                softmax[i] = 0
                softmax /= softmax.sum()

                # Calc p_i
                yi_mask = np.where(y == yi)
                p_i = softmax[yi_mask].sum()

                # Gradient computing
                sum_p = sum_m = 0.0
                outers_i = calc_outers_i(X, outers, i)

                for k in xrange(n):
                    s = softmax[k] * outers_i[k]
                    sum_p += s
                    if(yi == y[k]):
                        sum_m -= s

                grad += p_i * sum_p + sum_m
                succ += p_i

            succ /= len(y)

            update = True
            if self.adaptive_:
                if succ > succ_prev:
                    self.eta *= self.learn_inc
                else:
                    self.eta *= self.learn_dec
                    update = False
                    if self.eta < self.eta_thres:
                        stop = True

                succ_prev = succ

            if update:
                grad = 2 * L.dot(grad)
                L += self.eta * grad
                grad_norm = np.max(np.abs(grad))
                if grad_norm < self.prec or self.eta * grad_norm < self.tol:  # Difference between two iterations is given by eta*grad
                    stop = True

            self.num_its_ += 1
            if self.num_its_ == self.max_iter:
                stop = True

        self.L_ = L

        return self

    @staticmethod
    def _shuffle(X, y, outers=None):
        rnd = np.random.permutation(len(y))
        X = X[rnd, :]
        y = y[rnd]
        if outers is not None:
            for i in xrange(len(y)):
                outers[:, i] = outers[rnd, i]
            for i in xrange(len(y)):
                outers[i, :] = outers[i, rnd]
            # outers = outers[rnd,:][:,rnd]
        else:
            outers = None

        return X, y, outers

    @staticmethod
    def _compute_expected_success(L, X, y, class_split_inds=None):
        n, d = X.shape
        Lx = L.dot(X.T).T # todo not sure about this
        success = 0.0
        # cdef int i, j, i_max
        # cdef np.ndarray softmax, Lxi, dists, dists_i, yi_mask, Lxij
        # cdef float c, pw, p_i
        dists = euclidean_distances(Lx)
        for i in range(len(y)):
            softmax = np.empty([n], dtype=float)
            Lxi = Lx[i]
            # Ldiff = Lxi - Lx
            dists_i = -dists[i, :]  # -np.diag(Ldiff.dot(Ldiff.T))  # TODO improve efficiency of dists_i
            dists_i[i] = -np.inf
            i_max = np.argmax(dists_i)
            c = dists_i[i_max]          # TODO all distances can reach -inf
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
