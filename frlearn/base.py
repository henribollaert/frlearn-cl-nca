"""Base classes"""

from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import signature

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin


class ModelFactory(ABC):

    def __init__(self, preprocessors=()):
        self.preprocessors = preprocessors

    @abstractmethod
    def __call__(self, X, *, preprocessors=(), **kwargs) -> ModelFactory.Model:
        preprocessors = self.preprocessors + preprocessors
        preprocessing_models = []
        for preprocessor in preprocessors:
            extra_kwargs = {k: v for k, v in kwargs.items() if k in signature(preprocessor.__call__).parameters}
            preprocessing_model = preprocessor(X, **extra_kwargs)
            X = preprocessing_model(X)
            preprocessing_models.append(preprocessing_model)
        model = self._construct(X, **kwargs)
        model.preprocessing_models = preprocessing_models
        return model

    @property
    def construct(self):
        return self.__call__

    @abstractmethod
    def _construct(self, X, **kwargs) -> ModelFactory.Model:
        model = self.Model.__new__(self.Model)
        model.n, model.m = model.shape = X.shape
        return model

    class Model(ABC):

        n: int
        m: int
        shape: tuple[int, ...]
        preprocessing_models: list

        def __len__(self):
            return self.n

        @abstractmethod
        def __call__(self, X, *args, **kwargs):
            for preprocessing_model in self.preprocessing_models:
                X = preprocessing_model(X)
            return self._query(X, *args, **kwargs)

        @abstractmethod
        def _query(self, X, *args, **kwargs):
            pass


class Unsupervised(ModelFactory):

    def __call__(self, X) -> Unsupervised.Model:
        return super().__call__(X, )

    def _construct(self, X) -> Unsupervised.Model:
        model = super()._construct(X)
        return model

    class Model(ModelFactory.Model):
        pass


class ClassSupervised(ModelFactory):

    def __call__(self, X, y) -> ClassSupervised.Model:
        return super().__call__(X, y=y)

    def _construct(self, X, y) -> ClassSupervised.Model:
        model = super()._construct(X, y=y)
        model.classes = np.unique(y)
        model.n_classes = len(model.classes)
        return model

    class Model(ModelFactory.Model):

        classes: np.array
        n_classes: int


class LabelSupervised(ModelFactory):

    def __call__(self, X, Y) -> LabelSupervised.Model:
        return super().__call__(X, Y=Y)

    def _construct(self, X, Y) -> LabelSupervised.Model:
        model = super()._construct(X, Y=Y)
        model.n_labels = Y.shape[1]
        return model

    class Model(ModelFactory.Model):

        n_labels: int


class Classifier(ModelFactory):

    class Model(ModelFactory.Model):

        def __call__(self, X):
            return super().__call__(X)

        @property
        def query(self):
            return self.__call__

        @abstractmethod
        def _query(self, X):
            pass


class DataDescriptor(Unsupervised, Classifier):
    class Model(Unsupervised.Model, Classifier.Model):
        pass


class MultiClassClassifier(ClassSupervised, Classifier):
    class Model(ClassSupervised.Model, Classifier.Model):
        pass


class MultiLabelClassifier(LabelSupervised, Classifier):
    class Model(LabelSupervised.Model, Classifier.Model):
        pass


class FeaturePreprocessor(ModelFactory):

    class Model(ModelFactory.Model):

        def __call__(self, X):
            return super().__call__(X)

        @property
        def transform(self):
            return self.__call__


class FeatureSelector(FeaturePreprocessor):

    class Model(FeaturePreprocessor.Model):

        selection: np.array

        def _query(self, X):
            return X[:, self.selection]


class SupervisedInstancePreprocessor(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, X, y):
        pass


def select_class(scores, abstention_threshold: float = -1, labels=None):
    """
    Convert an array of class scores into class predictions, by selecting the class with the highest score.
    If none of the scores is greater than `abstention_threshold`, a generic `other` class will be predicted.
    
    Parameters
    ----------
    scores : array shape=(n, n_classes, )
        Array of class scores. Scores should be values in `[0, 1]`
    abstention_threshold : float=-1
        Threshold to use for predicting one of the classes.
    labels : array shape={(n_classes, ), (n_classes + 1, )}, default=None
        Labels of the classes in `scores` to be used in the return array. The first label is used for abstention,
        it may be omitted if `abstention_threshold == 0`. If `None`, positions are used instead, with 0 used
        for abstention if `abstention_threshold >= 0`.

    Returns
    -------
    predictions : array shape=(n, )
        Class label for each query instance.
    
    """
    if abstention_threshold >= 0:
        scores = np.concatenate([np.broadcast_to(abstention_threshold, (len(scores), 1)), scores], axis=-1)
    predictions = np.argmax(scores, axis=-1)
    if labels is not None:
        predictions = labels[predictions]
    return predictions


def discretise(scores, threshold: float = 0.5, ):
    """
    Discretise an array of label scores in `[0, 1]` into discrete predictions in `{0, 1}`,
    by selecting all labels that score higher than `threshold`.

    Parameters
    ----------
    scores : array shape=(n, n_classes, )
        Array of class scores. Scores should be values in `[0, 1]`
    threshold : float=0.5
        Threshold to use for selecting labels.

    Returns
    -------
    predictions : array shape=(n, )
        Class label for each query instance.

    """
    return scores >= threshold


def probabilities_from_scores(scores):
    """
    Rescale an array of class scores into probabilities that sum to 1, by dividing each score by the total sum.
    If all scores are zero, probabilities are assigned equally (`1/n_classes`).

    Parameters
    ----------
    scores : array shape=(n, n_classes, )
        Array of class scores.

    Returns
    -------
    probabilities : array shape=(n, n_classes, )
        Array of class probabilities.

    """
    rows_0 = ~np.all(scores, axis=1)
    scores[rows_0] = 1
    return scores/np.sum(scores, axis=1, keepdims=True)


class FitPredictClassifier(BaseEstimator, ClassifierMixin, ):
    """
    Convenience class for using any classifier as a scikit-learn-style classifier with fit and predict methods.

    Parameters
    ----------
    classifier_or_class : type or MultiClassClassifier or MultiLabelClassifier}
        Either an initialised classifier, or a classifier class. If a class, will be initialised with
        all remaining positional and keyword arguments.
    """

    def __init__(
            self,
            classifier_or_class: type[MultiClassClassifier] | type[MultiLabelClassifier] | MultiClassClassifier | MultiLabelClassifier,
            *args, **kwargs):
        super().__init__()
        if isinstance(classifier_or_class, ModelFactory):
            self.classifier = classifier_or_class
        else:
            self.classifier = classifier_or_class(*args, **kwargs)

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : array shape=(n_instances, n_features, )
            Training data. If array or matrix, shape [n_samples, n_features],
            or [n_samples, n_samples] if metric='precomputed'.

        y : {array-like, sparse matrix}
            Target values of shape = [n_samples] or [n_samples, n_outputs]

        """
        self.model_ = self.classifier(X, y)
        return self

    def predict(self, X):
        """
        Predict the class labels for the instances in X.

        Parameters
        ----------
        X : array shape=(n_instances, n_features, )
            Query instances.

        Returns
        -------
        y : array shape=(n_instances, )
            Class label for each query instance.
        """
        scores = self.model_(X)

        if isinstance(self.classifier, MultiClassClassifier):
            return select_class(scores, labels=self.model_.classes)
        else:
            return discretise(scores)

    def predict_proba(self, X):
        """
        Calculate probability estimates for the instances in X.

        Parameters
        ----------
        X : array shape=(n_instances, n_features, )
            Query instances.

        Returns
        -------
        p : array shape=(n_instances, n_classes, )
            The class probabilities of the query instances. Classes are ordered
            by lexicographic order.
        """
        # normalise membership degrees into confidence scores
        scores = self.model_(X)
        return probabilities_from_scores(scores)
