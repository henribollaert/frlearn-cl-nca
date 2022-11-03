"""
This module contains help functions for the experiments of my thesis.

Functions:
    compare_measures
    get_dataset
    get_number_of_nominal_features
    test_pandas
    bold_max
"""
import os

import pandas as pd
import re
import numpy as np

from frlearn.uncategorised.weights import LinearWeights
from sklearn.metrics import balanced_accuracy_score
from frlearn.base import select_class
from frlearn.classifiers import FRNN
from frlearn.feature_preprocessors import RangeNormaliser
from frlearn.neighbour_search_methods import BallTree


def compare_measures(folder_path,
                     distances,
                     nr_of_folds=10,
                     k=3,
                     remove_cat=True,
                     weights=LinearWeights(),
                     normaliser=RangeNormaliser()):
    """
    This is a help function for comparing FRNN with different similarity relations on a dataset using cross-validation.
    Returns a 2D array containing the balanced accuracies of each measure on each fold of the dataset.
    -------

    """

    accuracies = []
    for fold in range(nr_of_folds):
        # get the train and test sets
        x_train, y_train = get_dataset(folder_path, f"{fold + 1}tra", remove_cat=remove_cat)
        x_test, y_test = get_dataset(folder_path, f"{fold + 1}tst", remove_cat=remove_cat)

        # apply normalisation to train and test sets based on train set
        rn = normaliser(x_train)
        x_train_n = rn(x_train)
        x_test_n = rn(x_test)

        # calculate the accuracies for FRNN with each measure on this fold
        fold_accuracies = []
        for measure in distances:
            if measure.can_apply(x_train_n, y_train) and measure.can_apply(x_test_n, y_test):
                # fit the measure to the training set
                measure.fit(x_train_n, y_train)
                # instantiate the FRNN classifier factory
                clf = FRNN(preprocessors=(),
                           nn_search=BallTree(),
                           dissimilarity=measure.get_metric(),
                           lower_k=k,
                           upper_k=k,
                           lower_weights=weights,
                           upper_weights=weights)
                # construct the model
                model = clf(x_train_n, y_train)
                # query on the test set
                scores = model(x_test_n)
                # select classes with the highest scores and calculate the accuracy.
                classes = select_class(scores, labels=model.classes)
                fold_accuracies.append(balanced_accuracy_score(y_test, classes))
            else:
                fold_accuracies.append(np.NaN)
        accuracies.append(fold_accuracies)
    return accuracies


def test_save(measures_to_test,
              datasets_folder,
              results_folder,
              excluded_sets=None,
              must_include=None,
              verbose=False,
              remove_cat=True,
              weights=LinearWeights(),
              normaliser=RangeNormaliser(),
              k=20,
              nr_of_folds=10
              ):
    for dataset_dir in datasets_folder.iterdir():
        if dataset_dir.name != ".gitignore":
            short_name = dataset_dir.name[:re.search(r'\d', dataset_dir.name).start()][:-1]
            if not (excluded_sets is not None and short_name in excluded_sets) and \
                    (must_include is None or short_name in must_include):
                if verbose:
                    print(short_name)
                # create the folder for the results on this dataset
                dataset_result_path = results_folder / short_name
                if not os.path.exists(dataset_result_path):
                    os.makedirs(dataset_result_path)
                for fold in range(nr_of_folds):
                    # create the folder for the results on this fold of the dataset
                    fold_result_path = dataset_result_path / f"fold{fold + 1}"
                    if not os.path.exists(fold_result_path):
                        os.makedirs(fold_result_path)

                    # get the train and test sets
                    x_train, y_train = get_dataset(dataset_dir, f"{fold + 1}tra", remove_cat=remove_cat)
                    x_test, y_test = get_dataset(dataset_dir, f"{fold + 1}tst", remove_cat=remove_cat)

                    # apply normalisation to train and test sets based on train set
                    rn = normaliser(x_train)
                    x_train_n = rn(x_train)
                    x_test_n = rn(x_test)

                    # run FRNN with each measure on this fold
                    for measure in measures_to_test:
                        # check that we do not yet have results for these parameters
                        if not (fold_result_path / f"{measure.get_name()}_fold{fold + 1}.dat").is_file():
                            # check that we can apply the measure to both train and test sets
                            if measure.can_apply(x_train_n, y_train) and measure.can_apply(x_test_n, y_test):
                                # fit the measure to the training set
                                measure.fit(x_train_n, y_train)
                                # instantiate the FRNN classifier factory
                                clf = FRNN(preprocessors=(),
                                           nn_search=BallTree(),
                                           dissimilarity=measure.get_metric(),
                                           lower_k=k,
                                           upper_k=k,
                                           lower_weights=weights,
                                           upper_weights=weights)
                                # construct the model
                                model = clf(x_train_n, y_train)
                                # query on the test set
                                scores = model(x_test_n)
                                # select classes with the highest scores
                                with open(fold_result_path / f"{measure.get_name()}_fold{fold + 1}.dat", 'w') as f:
                                    for item in select_class(scores, labels=model.classes):
                                        f.write(f"{item}\n")
                            else:
                                with open(fold_result_path / f"{measure.get_name()}_fold{fold + 1}.dat", 'w') as f:
                                    f.write("NaN\n")


def get_dataset(folder_path, keyword, remove_cat=True):
    """
    Returns a dataset from a specified folder with a given keyword in the name, possibly after removing categorical
    features.

    Parameters
    ----------
    folder_path
    keyword
    remove_cat

    Returns
    -------
    numpy array containing x values, numpy array containing y values
    """
    set_list = [_ for _ in folder_path.iterdir() if keyword in _.name]
    assert len(set_list) == 1, f'{ len(set_list)} files with {keyword} in their name.'

    dataset = pd.read_csv(set_list[0], header=None, comment='@')
    if remove_cat:
        nums = [t != 'object' for t in dataset.dtypes]
        nums[-1] = False
        x_dataset = dataset.loc[:, nums]
    else:
        x_dataset = dataset.iloc[:, :-1]
    y_dataset = dataset.iloc[:, -1]
    return x_dataset.values, y_dataset.values


def get_number_of_nominal_features(folder):
    d = {}
    for dataset_dir in folder.iterdir():
        if dataset_dir.name != ".gitignore":
            short_name = dataset_dir.name[:re.search(r'\d', dataset_dir.name).start()][:-1]
            dataset = pd.read_csv([_ for _ in dataset_dir.iterdir() if '1tra' in _.name][0], header=None, comment='@')
            cats = 0
            for t in dataset.dtypes[:-1]:
                if t == 'object':
                    cats += 1
            # cats = len([t == 'object' for t in dataset.dtypes])
            d[short_name] = cats
    return d


def test_pandas(measures_to_test, datasets_folder, excluded_sets=None, must_include=None, verbose=False,
                remove_cat=True, weights=LinearWeights(), normaliser=RangeNormaliser(), k=20):
    frame = pd.DataFrame(columns=[d.get_name() for d in measures_to_test])
    for dataset_dir in datasets_folder.iterdir():
        if dataset_dir.name != ".gitignore":
            short_name = dataset_dir.name[:re.search(r'\d', dataset_dir.name).start()][:-1]
            if not (excluded_sets is not None and short_name in excluded_sets) and \
                    (must_include is None or short_name in must_include):
                if verbose:
                    print(short_name)
                means = np.mean(compare_measures(dataset_dir,
                                                 measures_to_test,
                                                 remove_cat=remove_cat,
                                                 weights=weights,
                                                 normaliser=normaliser,
                                                 k=k),
                                axis=0)
                if verbose:
                    print(means)
                frame.loc[short_name] = means
    return frame


def bold_max(data, format_string="%.3f"):
    """
    Returns a pandas dataframe with formatted strings and bolded maximaL values.

    Parameters
    ----------
    data
    format_string

    Returns
    -------

    """
    maxima = data != data.max()
    bolded = data.apply(lambda x: "\\textbf{%s}" % format_string % x)
    formatted = data.apply(lambda x: format_string % x)
    return formatted.where(maxima, bolded)


def calculate_score(data_folder,
                    results_folder,
                    metric,
                    wanted_measures,
                    excluded_sets=None,
                    nr_of_folds=10,
                    verbose=False):
    if excluded_sets is None:
        excluded_sets = ['abalone']
    frame = pd.DataFrame(columns=wanted_measures)
    for dataset_dir in data_folder.iterdir():
        if dataset_dir.name != ".gitignore":
            # very dirty, search depends on the fact that the names don't contain underscores
            short_name = dataset_dir.name[:re.search(r'\d', dataset_dir.name).start()][:-1]
            if short_name not in excluded_sets:
                if verbose:
                    print(short_name)
                dataset_result_path = results_folder / short_name

                # create dictionaries to save results of this dataset
                sum_of_metrics = {}
                successful_results = {}
                for fold in range(nr_of_folds):
                    # create the folder for the results on this fold of the dataset
                    fold_result_path = dataset_result_path / f"fold{fold + 1}"
                    if fold_result_path.exists():
                        _, y_test = get_dataset(dataset_dir, f"{fold + 1}tst")
                        for measure_result in [_ for _ in fold_result_path.iterdir()]:
                            measure_name = measure_result.name[:re.search(r'_', measure_result.name).start()]
                            predictions = pd.read_csv(measure_result, header=None)
                            if not predictions.isnull().values.any():
                                if measure_name not in successful_results:
                                    successful_results[measure_name] = 1
                                    sum_of_metrics[measure_name] = metric(y_test, predictions)
                                else:
                                    successful_results[measure_name] += 1
                                    sum_of_metrics[measure_name] += metric(y_test, predictions)

                # add scores to pandas dataframe
                for s in wanted_measures:
                    frame.at[short_name, s] = \
                        sum_of_metrics[s]/successful_results[s] if s in sum_of_metrics.keys() else np.NaN
                    print(short_name, successful_results)
    return frame
