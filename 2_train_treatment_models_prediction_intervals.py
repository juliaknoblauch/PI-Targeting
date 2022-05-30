# NOTE: I use a development version of treat-learn
# For general version:
# https://github.com/johaupt/treatment-learn

import multiprocessing as mp
from copy import copy
import itertools
import collections
import time
from datetime import date

import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, \
    RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.model_selection._search import ParameterGrid
from collections import defaultdict
from collections import OrderedDict
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import make_scorer
from pprint import pprint
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import RepeatedStratifiedKFold
import helper
from helper import *

from typing import Union
from typing_extensions import TypedDict
from mapie.regression import MapieRegressor
from mapie.subsample import Subsample
from nonconformist.nc import QuantileRegAsymmetricErrFunc
from sklearn.ensemble import RandomForestRegressor
from nonconformist.nc import RegressorNormalizer
from nonconformist.nc import AbsErrorErrFunc
from nonconformist.cp import IcpRegressor
from xbcausalforest import XBCF

from treatlearn.double_robust_transformation import DoubleRobustTransformer
from treatlearn.indirect import SingleModel, HurdleModel, TwoModelRegressor
from treatlearn.evaluation import transformed_outcome_loss
from myXBCF import myXBCF


def QR_CV(model, X, y, alpha, group, param_grid):
    """
    Tunes the hyperparamers of QR with pinball loss under HalvingRandomSearchCV
    Parameters
     ----------
    X : numpy array
        scaled covariates of observations
    y: numpy array,
        true labels (n)
    group: int
        treatment group assignment, 1: treatment, 0: control
   param_grid: dict
        paramamter grid to tune the

    Returns
    -------
    best_params_: dict
        Parameter setting that gave the best results on the hold out data.
    """
    neg_mean_pinball_loss_scorer = make_scorer(
        mean_pinball_loss,
        alpha=alpha,
        greater_is_better=False,  # maximize the negative loss
    )
    if group == 'single':

        search_alpha = HalvingRandomSearchCV(
            model,
            param_grid['gbtr'],
            resource="n_estimators",
            max_resources=250,
            min_resources=50,
            scoring=neg_mean_pinball_loss_scorer,
            n_jobs=1,
            random_state=0).fit(X, y)

    elif group == 'dr':

        search_alpha = HalvingRandomSearchCV(
            model,
            param_grid['gbtr'],
            resource="n_estimators",
            max_resources=250,
            min_resources=50,
            scoring=neg_mean_pinball_loss_scorer,
            n_jobs=1,
            random_state=0).fit(X, y)

    else:
        search_alpha = HalvingRandomSearchCV(
            model,
            param_grid['gbtr'],
            resource="n_estimators",
            max_resources=250,
            min_resources=50,
            scoring=neg_mean_pinball_loss_scorer,
            n_jobs=1,
            random_state=0).fit(X[g == group, :], y[g == group])

    print(search_alpha.best_params_)
    return search_alpha.best_params_


def get_train_validation_test_split(folds, val_set=True):
    # Get only test indices per fold
    folds_test = [fold[1] for fold in folds]

    # Get union of indices
    indices = np.concatenate(folds_test)
    print(len(indices))
    print(len(np.unique(indices)))
    # if len(indices) != len(np.unique(indices)): raise ValueError("Overlap in test indices")

    # Get validation indices per fold by taking next test fold
    if val_set:
        folds_validation = collections.deque(folds_test)
        folds_validation.rotate(-1)
        folds_validation = list(folds_validation)
    else:
        folds_validation = [np.array([]) for fold in folds_test]

    # Get training indices per fold from remaining
    folds_train = [np.setdiff1d(indices, np.concatenate([test, val])) for test, val in
                   zip(folds_test, folds_validation)]

    # Put folds together
    output = [{'train': train, 'val': val, 'test': test} for train, val, test in
              zip(folds_train, folds_validation, folds_test)]

    return output


def predict_treatment_models(X, y, c, g, tau_conversion, tau_basket, tau_response, split, fold_index, TUNE_CATE=False,
                             TUNE_PI=False,
                             Jackknife=True, qr=True, CQR=True, xbcf=True):
    print(f"Working on fold {fold_index}:")

    treatment_model_lib = {}
    conversion_model_lib = {}
    hyperparameters = {}
    time_model = {}
    time_pi = {}

    N_JOBS = 1

    # Find columns that are not binary with max!=1
    num_columns = np.where(X.columns[(X.max(axis=0) != 1)])[0].tolist()
    n_cat = (X.max(axis=0) == 1).sum()
    # Fixed parameters for TUNE = False
    params = {
        "gbtr_control":
            {"learning_rate": 0.05,
             "n_estimators": 100,
             "max_depth": 3,
             "subsample": 0.95,
             'min_samples_leaf': 100,
             },
        "gbtr":
            {"learning_rate": 0.1,
             "n_estimators": 100,
             "max_depth": 4,
             "subsample": 0.95,
             'min_samples_leaf': 100,
             },
        "gbtr_2ndstage":
            {"learning_rate": 0.075,
             "n_estimators": 100,
             "max_depth": 4,
             "subsample": 0.95,
             "max_features": 0.9,
             'min_samples_leaf': 50,
             },
        "gbtc":
            {"learning_rate": 0.1,
             "n_estimators": 100,
             "max_depth": 4,
             "subsample": 0.95,
             'min_samples_leaf': 100,
             },
        "rf": {
            "n_estimators": 500,
            "min_samples_leaf": 50,
        },
        "reg": {
            "alpha": 10
        },
        "logit": {
            "C": 1,
            'solver': 'liblinear',
            'max_iter': 1000
        },
        'xbcf': {
            'num_sweeps': 100,
            'burnin': 20,
            'num_trees_pr': 100,
            'num_trees_trt': 50,
            'num_cutpoints': 100,
            'alpha_pr': 0.95,
            'beta_pr': 2,
            'alpha_trt': 0.95,
            'beta_trt': 2,

        },
        "gbtr_single":
            {"learning_rate": 0.15,
             "n_estimators": 100,
             "max_depth": 4,
             "subsample": 0.95,
             'min_samples_leaf': 100,
             },
        "two-model_outcome_gbtr":
            {"learning_rate": 0.075,
             "n_estimators": 100,
             "max_depth": 2,
             "subsample": 0.95,
             'min_samples_leaf': 100,
             },
        "two-model_hurdle_gbtc":
            {"learning_rate": 0.05,
             "n_estimators": 100,
             "max_depth": 2,
             "subsample": 0.95,
             'min_samples_leaf': 50,
             },
        "two-model_hurdle_gbt_gbtr_2ndstage":
            {"learning_rate": 0.05,
             "n_estimators": 100,
             "max_depth": 2,
             "subsample": 0.95,
             'min_samples_leaf': 50,
             },
        "DR_gbtr":
            {"learning_rate": 0.05,
             "n_estimators": 100,
             "max_depth": 2,
             "subsample": 0.95,
             'min_samples_leaf': 100,
             },
    }
    # Param Tuning if Tuning = True
    param_grid = {
        'gbtr': {
            'learning_rate': [0.05, 0.075, 0.1, 0.125, 0.15],
            'max_depth': [2, 3, 4],
            # 'n_estimators': [100],
            'subsample': [0.95],
            # 'max_features':[0.9],
            'min_samples_leaf': [1, 50, 100],
        },
        'gbtc': {
            'learning_rate': [0.05, 0.075, 0.1, 0.125, 0.15],
            'max_depth': [2, 3, 4],
            'n_estimators': [100],
            'subsample': [0.95],
            # 'max_features':[0.9],
            'min_samples_leaf': [1, 50, 100],
        },
        'gbtr_2ndstage': {
            'learning_rate': [0.05, 0.075, 0.1, 0.125, 0.15],
            'max_depth': [2, 3, 4],
            'n_estimators': [100],
            'subsample': [0.95],
            # 'max_features':[0.9],
            'min_samples_leaf': [1, 50, 100],
        },
        'gbtr_control': {
            'learning_rate': [0.05, 0.075, 0.1, 0.125, 0.15],
            'max_depth': [2, 3, 4],
            'n_estimators': [100],
            'subsample': [0.95],
            # 'max_features':[0.9],
            'min_samples_leaf': [1, 50, 100],
        },
        'rf': {
            'n_estimators': [500],
            'min_samples_leaf': [50],
            'max_features': [0.05, 0.1, 0.15],
        },
        "reg": {
            "alpha": [10, 2, 1, 0.5, 0.25, 0.166, 0.125, 0.1]
        },
        "logit": {
            "C": [0.1, 0.5, 1, 2, 4, 6, 8, 10],
            'solver': ['liblinear'],
            'max_iter': [1000]
        },
        "xbcf": {
            #### for standard see fixed params above
            'num_sweeps': [100],
            'burnin': [20],
            'num_trees_pr': [100],
            'num_trees_trt': [50],
            'num_cutpoints': [100],
            'alpha_pr': [0.95],
            'beta_pr': [2],
            'alpha_trt': [0.95],
            'beta_trt': [2],  # optional: add 1.5, instead of only 2
            #####
            'mtry_pr': [int(X.shape[1])],
            'mtry_trt': [int(X.shape[1])],
            'p_categorical_pr': [int(n_cat)],
            'p_categorical_trt': [int(n_cat)],
            'tau_pr': [0.4, 0.6],  # default actually
            'tau_trt': [0.05, 0.1],
            # 'num_trees': [20,30],
            # params['xbcf']['tau_pr'] = 0.1/params['xbcf']['num_trees_pr'] # 0.1 * var(y_norm) = 0.1
            # params['xbcf']['tau_trt'] = 0.1/params['xbcf']['num_trees_trt']

        }}

    # Split the train and validation data
    X_test, y_test, c_test, g_test, tau_conversion_test, tau_basket_test, tau_response_test = [
        obj.to_numpy().astype(float)[split['test']] for obj in [X, y, c, g, tau_conversion, tau_basket, tau_response]]
    # X_val, y_val, c_val, g_val, tau_conversion_val, tau_basket_val, tau_response_val = [obj.to_numpy().astype(float)[split['val']] for obj in [X, y, c, g, tau_conversion, tau_basket, tau_response]]
    X, y, c, g, tau_conversion, tau_basket, tau_response = [obj.to_numpy().astype(float)[split['train']] for obj in
                                                            [X, y, c, g, tau_conversion, tau_basket, tau_response]]

    # Normalize the data
    ct = ColumnTransformer([
        # (name, transformer, columns)
        # Transformer for categorical variables
        # ("onehot",
        #     OneHotEncoder(categories='auto', handle_unknown='ignore', ),
        #     cat_columns),
        # Transformer for numeric variables
        ("num_preproc", StandardScaler(), num_columns)
    ],
        remainder="passthrough")

    X = ct.fit_transform(X)
    X_test = ct.transform(X_test)
    # X_val = ct.transform(X_val)

    # save the original target values
    y_orig = y
    y_test_orig = y_test
    # y_val_orig = y_val

    # Treatment indicator as variable
    Xg = np.c_[X, g]

    Xg_test = np.c_[X_test, g_test]
    # Xg_test = np.c_[X_val, g_val]
    yg = np.c_[y, g]
    yg = np.c_[y, g]

    # Double robust transformation
    dr = DoubleRobustTransformer()
    y_dr = dr.fit_transform(X, y, g)

    #### Parameter Tuning  ####
    # Cross-validation folds stratified randomization by (outcome x treatment group)
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    cg_groups = 2 * g + c  # Groups 0-4 depending on combinations [0,1]x[0,1]
    folds = list(splitter.split(X, cg_groups))
    folds_c1 = list(splitter.split(X[c == 1, :], g[c == 1]))

    ## Simple GBT predictors
    # Tune estimator of spending
    if TUNE_CATE == True:
        cv = GridSearchCV(GradientBoostingRegressor(), param_grid["gbtr"], scoring='neg_mean_squared_error',
                          n_jobs=N_JOBS,
                          verbose=0, cv=folds)
        cv.fit(X, y)
        params["gbtr"] = cv.best_params_
        print(f"gbtr params: {cv.best_params_}")
        # pickle.dump(params, open(f"parameters/{fold_index}_save_{today}.p", "wb"))

    # # Tune estimator of conversion
    if TUNE_CATE == True:
        cv = GridSearchCV(GradientBoostingClassifier(), param_grid["gbtc"], scoring='neg_brier_score', n_jobs=N_JOBS,
                          verbose=0, cv=folds)
        cv.fit(X, c)
        params["gbtc"] = cv.best_params_
        print(f"gbtc params: {cv.best_params_}")

        # pickle.dump(params, open(f"parameters/{fold_index}_save_{today}.p", "wb"))
    #
    # # Tune estimator of spending given conversion
    if TUNE_CATE == True:
        cv = GridSearchCV(GradientBoostingRegressor(), param_grid["gbtr"], scoring='neg_mean_squared_error',
                          n_jobs=N_JOBS, verbose=0, cv=folds_c1)
        cv.fit(X[c == 1, :], y[c == 1])
        params["gbtr_2ndstage"] = cv.best_params_
        print(f"gbtr_2ndstage params: {cv.best_params_}")

    params['xbcf']['mtry_pr'] = int(X.shape[1])
    params['xbcf']['mtry_trt'] = int(X.shape[1])
    params['xbcf']['p_categorical_pr'] = int(n_cat)
    params['xbcf']['p_categorical_trt'] = int(n_cat)
    # print(f"xbcf default params: {params['xbcf']}")

    ######## CATE ########

    #### Single Model ####
    ## GBT regression
    start = time.time()
    single_gbt_regressor = SingleModel(GradientBoostingRegressor(**params["gbtr_single"]))

    if TUNE_CATE == True:
        # Tune single model based on Transformed Outcome Loss
        best_params = grid_search_cv(X, y, g, single_gbt_regressor, param_grid['gbtr'], folds)
        single_gbt_regressor.set_params(**best_params)
        hyperparameters['single_gbt_regressor'] = best_params
        print(f"single_gbt_regressor params: {best_params}")

    single_gbt_regressor.fit(X, y, g=g)
    time_model['single-model_outcome_gbt'] = time.time() - start
    treatment_model_lib['single-model_outcome_gbt'] = single_gbt_regressor

    ## Hurdle Gradient Boosting
    start = time.time()
    single_hurdle_gbt = SingleModel(HurdleModel(conversion_classifier=GradientBoostingClassifier(**params["gbtc"]),
                                                value_regressor=GradientBoostingRegressor(**params["gbtr_2ndstage"])))
    # if TUNE_CATE == True:
    #    best_params = grid_search_cv(X=X, y=y, g=g, c=c, estimator=single_hurdle_gbt, param_grid=param_grid['gbtr'], folds=folds)
    #    print(f"single model Hurdle GBT params: {best_params}")
    #    # -> Tuning while fixing same parameters for both models of the hurdle does not give good results
    #    # -> Using optimal parameter tuned for outcome prediction instead

    single_hurdle_gbt.fit(X=X, y=y, c=c, g=g)
    time_model['single-model_hurdle_gbt'] = time.time() - start
    treatment_model_lib['single-model_hurdle_gbt'] = single_hurdle_gbt

    # #### Two-Model Approach ####

    # # ## Gradient Boosting Regression
    start = time.time()
    two_model_outcome_gbt = TwoModelRegressor(
        control_group_model=GradientBoostingRegressor(**params["two-model_outcome_gbtr"]),
        treatment_group_model=GradientBoostingRegressor(**params["two-model_outcome_gbtr"]))
    #
    if TUNE_CATE == True:
        best_params = grid_search_cv(X, y, g, two_model_outcome_gbt, param_grid['gbtr'], folds)
        two_model_outcome_gbt.set_params(**best_params)
        print(f"Two-Model outcome GBT params: {best_params}")
        hyperparameters['two-model_outcome_gbt'] = best_params

    #
    treatment_model_lib["two-model_outcome_gbt"] = two_model_outcome_gbt.fit(X=X, y=y, g=g)
    time_model["two-model_outcome_gbt"] = time.time() - start

    #
    # ## Hurdle GBT
    start = time.time()
    two_model_hurdle_gbt = TwoModelRegressor(control_group_model=HurdleModel(
        conversion_classifier=GradientBoostingClassifier(**params["two-model_hurdle_gbtc"]),
        value_regressor=GradientBoostingRegressor(**params["two-model_hurdle_gbt_gbtr_2ndstage"])),
        treatment_group_model=HurdleModel(
            conversion_classifier=GradientBoostingClassifier(**params["two-model_hurdle_gbtc"]),
            value_regressor=GradientBoostingRegressor(**params["two-model_hurdle_gbt_gbtr_2ndstage"])
        ))
    if TUNE_CATE == True:
        param_grid_two_model_hurdle = {
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [2, 3, 4],
            'n_estimators': [100],
            'subsample': [0.95],
            # 'max_features':[0.9],
            'min_samples_leaf': [50],
        }
        best_params = grid_search_cv_hurdle(X=X, y=y, g=g, c=c, estimator=two_model_hurdle_gbt,
                                            param_grid_conversion=param_grid_two_model_hurdle,
                                            param_grid_regression=param_grid_two_model_hurdle,
                                            folds=folds)
        for model in [two_model_hurdle_gbt.treatment_group_model, two_model_hurdle_gbt.control_group_model]:
            model.conversion_classifier.set_params(**best_params[0])
            model.value_regressor.set_params(**best_params[1])

        print(f"Two-Model hurdle GBT params: {best_params}")
        hyperparameters['two-model_hurdle_gbt'] = best_params

    treatment_model_lib["two-model_hurdle_gbt"] = two_model_hurdle_gbt.fit(X=X, y=y, g=g, c=c)
    time_model["two-model_hurdle_gbt"] = time.time() - start

    #### Double robust ####
    ## Regression  with GBT ##
    start = time.time()

    if TUNE_CATE == True:
        cv = GridSearchCV(GradientBoostingRegressor(), param_grid['gbtr'], scoring='neg_mean_squared_error',
                          n_jobs=N_JOBS, verbose=0, cv=folds)
        cv.fit(X, y_dr)
        print(f"DR GBT params: {cv.best_params_}")
        treatment_model_lib["dr_outcome_gbt"] = GradientBoostingRegressor().set_params(**cv.best_params_)
        hyperparameters['dr_outcome_gbt'] = best_params

    else:
        treatment_model_lib["dr_outcome_gbt"] = GradientBoostingRegressor(**params["DR_gbtr"])

    treatment_model_lib["dr_outcome_gbt"].fit(X, y_dr)
    time_model['dr_outcome_gbt'] = time.time() - start

    #### XBCF ####
    # hyperparameter tuning off since takes too long and no improvement
    if xbcf == True:
        start = time.time()
        xbcf_model = myXBCF(standardize_target=True, **params["xbcf"])

        # optional parameter tuning for XBCF based on TOL, not used since computationally complex
        # param_opt = False
        # if param_opt == True:
        #     best_params, results = grid_search_cv_xbcf(X,X, y, g, xbcf_model, param_grid['xbcf'], folds)
        #     print('best params were found and our results are')
        #     hyperparameters['xbcf_outcome_xbcf'] = best_params
        #     hyperparameters['results'] = results
        #     print(hyperparameters)
        #     xbcf_model = myXBCF(standardize_target=True, **best_params)

        xbcf_model.fit(x_t=X, x=X, y=y, z=g.astype('int32'))
        time_model['xbcf_outcome_xbcf'] = time.time() - start

        treatment_model_lib["xbcf_outcome_xbcf"] = xbcf_model
        print('Finished XBCF.')
        hyperparameters['xbcf_outcome_xbcf'] = treatment_model_lib["xbcf_outcome_xbcf"].get_params()

    ##### Conversion Models ####

    conversion_model_lib["single-model_outcome_gbt"] = GradientBoostingClassifier(**params["gbtc"])
    conversion_model_lib["single-model_outcome_gbt"].fit(X[g == 1, :], c[g == 1])

    ### Evaluation
    print('Starting Prediction of CATE')
    ##### Conversion treatment effect

    treatment_conversion_train = {}
    treatment_conversion_test = {}

    treatment_conversion_train["single-model_hurdle_gbt"] = treatment_model_lib[
                                                                'single-model_hurdle_gbt'].model.conversion_classifier.predict_proba(
        np.c_[X, np.ones((X.shape[0], 1))])[:, 1] - treatment_model_lib[
                                                        'single-model_hurdle_gbt'].model.conversion_classifier.predict_proba(
        np.c_[X, np.zeros((X.shape[0], 1))])[:, 1]
    treatment_conversion_test["single-model_hurdle_gbt"] = treatment_model_lib[
                                                               'single-model_hurdle_gbt'].model.conversion_classifier.predict_proba(
        np.c_[X_test, np.ones((X_test.shape[0], 1))])[:, 1] - treatment_model_lib[
                                                                  'single-model_hurdle_gbt'].model.conversion_classifier.predict_proba(
        np.c_[X_test, np.zeros((X_test.shape[0], 1))])[:, 1]

    treatment_conversion_train["two-model_hurdle_gbt"] = treatment_model_lib[
                                                             'two-model_hurdle_gbt'].treatment_group_model.predict_hurdle(
        X) - treatment_model_lib['two-model_hurdle_gbt'].control_group_model.predict_hurdle(X)
    treatment_conversion_test["two-model_hurdle_gbt"] = treatment_model_lib[
                                                            'two-model_hurdle_gbt'].treatment_group_model.predict_hurdle(
        X_test) - treatment_model_lib['two-model_hurdle_gbt'].control_group_model.predict_hurdle(X_test)
    #
    treatment_conversion_train["ATE__"] = (c[g == 1].mean()) - (c[g == 0].mean()) * np.ones([X.shape[0]])
    treatment_conversion_test["ATE__"] = (c[g == 1].mean()) - (c[g == 0].mean()) * np.ones([X_test.shape[0]])

    # Baselines
    treatment_conversion_train["oracle__"] = tau_conversion
    treatment_conversion_test["oracle__"] = tau_conversion_test

    # ##### Basket value treatment effect

    treatment_basketvalue_train = {}
    treatment_basketvalue_test = {}

    treatment_basketvalue_train["single-model_hurdle_gbt"] = treatment_model_lib[
                                                                 'single-model_hurdle_gbt'].model.value_regressor.predict(
        np.c_[X, np.ones((X.shape[0], 1))]) - treatment_model_lib[
                                                                 'single-model_hurdle_gbt'].model.value_regressor.predict(
        np.c_[X, np.zeros((X.shape[0], 1))])
    treatment_basketvalue_test["single-model_hurdle_gbt"] = treatment_model_lib[
                                                                'single-model_hurdle_gbt'].model.value_regressor.predict(
        np.c_[X_test, np.ones((X_test.shape[0], 1))]) - treatment_model_lib[
                                                                'single-model_hurdle_gbt'].model.value_regressor.predict(
        np.c_[X_test, np.zeros((X_test.shape[0], 1))])

    treatment_basketvalue_train["two-model_hurdle_gbt"] = treatment_model_lib[
                                                              'two-model_hurdle_gbt'].treatment_group_model.predict_value(
        X) - treatment_model_lib['two-model_hurdle_gbt'].control_group_model.predict_value(X)
    treatment_basketvalue_test["two-model_hurdle_gbt"] = treatment_model_lib[
                                                             'two-model_hurdle_gbt'].treatment_group_model.predict_value(
        X_test) - treatment_model_lib['two-model_hurdle_gbt'].control_group_model.predict_value(X_test)
    #
    treatment_basketvalue_train["ATE__"] = (y_orig[(c == 1) & (g == 1)].mean()) - (
        y_orig[(c == 1) & (g == 0)].mean()) * np.ones([X.shape[0]])
    treatment_basketvalue_test["ATE__"] = (y_orig[(c == 1) & (g == 1)].mean()) - (
        y_orig[(c == 1) & (g == 0)].mean()) * np.ones([X_test.shape[0]])

    treatment_basketvalue_train["oracle__"] = tau_basket
    treatment_basketvalue_test["oracle__"] = tau_basket_test

    # ##### Treatment response prediction

    treatment_pred_train = {key: model.predict(X) for key, model in treatment_model_lib.items()}
    treatment_pred_test = {key: model.predict(X_test) for key, model in treatment_model_lib.items()}
    # treatment_pred_val = {key: model.predict(X_val) for key, model in treatment_model_lib.items()}

    # Baselines
    treatment_pred_train["oracle__"] = tau_response
    treatment_pred_test["oracle__"] = tau_response_test
    # treatment_pred_val["oracle__"] =   tau_response_val

    treatment_pred_train["ATE__"] = (y[g == 1].mean()) - (y[g == 0].mean()) * np.ones([X.shape[0]])
    treatment_pred_test["ATE__"] = (y[g == 1].mean()) - (y[g == 0].mean()) * np.ones([X_test.shape[0]])
    # treatment_pred_val["ATE__"] =   (y[g==1].mean())-(y[g==0].mean()) * np.ones([X_val.shape[0]])

    ############ Conversion C(T=1) prediction
    conversion_pred_train = {key: model.predict_proba(X)[:, 1] for key, model in conversion_model_lib.items()}
    conversion_pred_test = {key: model.predict_proba(X_test)[:, 1] for key, model in conversion_model_lib.items()}

    conversion_pred_train["single-model_hurdle_gbt"] = treatment_model_lib[
                                                           'single-model_hurdle_gbt'].model.conversion_classifier.predict_proba(
        np.c_[X, np.ones((X.shape[0], 1))])[:, 1]
    conversion_pred_test["single-model_hurdle_gbt"] = treatment_model_lib[
                                                          'single-model_hurdle_gbt'].model.conversion_classifier.predict_proba(
        np.c_[X_test, np.ones((X_test.shape[0], 1))])[:, 1]
    #
    conversion_pred_train["two-model_hurdle_gbt"] = treatment_model_lib[
                                                        "two-model_hurdle_gbt"].treatment_group_model.conversion_classifier.predict_proba(
        X)[:, 1]
    conversion_pred_test["two-model_hurdle_gbt"] = treatment_model_lib[
                                                       "two-model_hurdle_gbt"].treatment_group_model.conversion_classifier.predict_proba(
        X_test)[:, 1]

    conversion_pred_train["Conversion-Rate__"] = np.ones(X.shape[0]) * c[g == 1].mean()
    conversion_pred_test["Conversion-Rate__"] = np.ones(X_test.shape[0]) * c[g == 1].mean()

    print('Finished CATE Prediction.')

    ######## Prediction Interval Estimation ########
    print('Start PI Estimation.')
    prediction_intervals_train = {}
    prediction_intervals_test = {}

    ### Uncertainty / Prediction Intervals #####
    alpha_list = [0.05, 0.32]
    for alpha in alpha_list:
        print(f'Start PI Estimation for {alpha}.')
        LOWER_ALPHA = alpha / 2
        UPPER_ALPHA = 1 - alpha / 2

        prediction_intervals_train[alpha] = defaultdict(dict)
        prediction_intervals_test[alpha] = defaultdict(dict)
        time_pi[alpha] = defaultdict(dict)

        #### XBCF ####
        if xbcf == True:
            test_pred_low, test_pred_high = treatment_model_lib["xbcf_outcome_xbcf"].predict_prediction_interval(X_test,
                                                                                                                 LOWER_ALPHA,
                                                                                                                 UPPER_ALPHA,
                                                                                                                 burnin_off=False)
            train_pred_low, train_pred_high = treatment_model_lib["xbcf_outcome_xbcf"].predict_prediction_interval(X,
                                                                                                                   LOWER_ALPHA,
                                                                                                                   UPPER_ALPHA,
                                                                                                                   burnin_off=False)

            prediction_intervals_train[alpha]['xbcf_outcome_xbcf']['quantile_model'] = {}
            prediction_intervals_test[alpha]['xbcf_outcome_xbcf']['quantile_model'] = {}
            prediction_intervals_test[alpha]['xbcf_outcome_xbcf']['quantile_model'][
                'pred_low'] = test_pred_low
            prediction_intervals_test[alpha]['xbcf_outcome_xbcf']['quantile_model'][
                'pred_high'] = test_pred_high
            prediction_intervals_train[alpha]['xbcf_outcome_xbcf']['quantile_model'][
                'pred_low'] = train_pred_low
            prediction_intervals_train[alpha]['xbcf_outcome_xbcf']['quantile_model'][
                'pred_high'] = train_pred_high
            time_pi[alpha]['xbcf_outcome_xbcf'] = time_model['xbcf_outcome_xbcf']

            print('Finished XBCF.')

        #### Quantile Regression ####
        print('Start Quantile Regression.')
        # For model agnostic approoach --> alpha/2 in treatment/control group

        if qr == True:
            ##0 Tune Hyperparameters ##
            # note that tuning was run for subset of converted customers (not explicit here) and later merged to the
            # other data since the individuals in the folds are the same
            # TREATMENT
            # upper quantile
            start = time.time()
            quantile_model_high_treat = GradientBoostingRegressor(loss='quantile', alpha=UPPER_ALPHA / 2)
            if TUNE_PI == True:

                start_tune1 = time.time()
                best_params_ = QR_CV(model=quantile_model_high_treat, X=X, y=y, alpha=UPPER_ALPHA / 2, group=1,
                                     param_grid=param_grid)
                quantile_model_high_treat = GradientBoostingRegressor(loss='quantile', alpha=UPPER_ALPHA / 2,
                                                                      **best_params_)
                time_tune1_high = time.time() - start_tune1
                hyperparameters[f'{UPPER_ALPHA / 2}_1'] = best_params_

            quantile_model_high_treat.fit(X[g == 1, :], y[g == 1])  # , c[g == 1]

            # lower quantile
            quantile_model_low_treat = GradientBoostingRegressor(loss='quantile', alpha=LOWER_ALPHA / 2)
            if TUNE_PI == True:
                start_tune1 = time.time()
                best_params_ = QR_CV(model=quantile_model_low_treat, X=X, y=y, alpha=LOWER_ALPHA / 2, group=1,
                                     param_grid=param_grid)
                quantile_model_low_treat = GradientBoostingRegressor(loss='quantile', alpha=LOWER_ALPHA / 2,
                                                                     **best_params_)
                time_tune1_low = time.time() - start_tune1
                hyperparameters[f'{LOWER_ALPHA / 2}_1'] = best_params_

            quantile_model_low_treat.fit(X[g == 1, :], y[g == 1])
            time_treat = time.time() - start

            # CONTROL
            # upper quantile
            start = time.time()
            quantile_model_high_control = GradientBoostingRegressor(loss='quantile', alpha=UPPER_ALPHA / 2)
            if TUNE_PI == True:
                start_tune0 = time.time()
                best_params_ = QR_CV(model=quantile_model_high_control, X=X, y=y, alpha=UPPER_ALPHA / 2, group=0,
                                     param_grid=param_grid)
                time_tune0_high = time.time() - start_tune0
                quantile_model_high_control = GradientBoostingRegressor(loss='quantile', alpha=UPPER_ALPHA / 2,
                                                                        **best_params_)
                hyperparameters[f'{UPPER_ALPHA / 2}_0'] = best_params_
            quantile_model_high_control.fit(X[g == 0, :], y[g == 0])

            # lower quantile
            quantile_model_low_control = GradientBoostingRegressor(loss='quantile', alpha=LOWER_ALPHA / 2)
            if TUNE_PI == True:
                start_tune0 = time.time()
                best_params_ = QR_CV(model=quantile_model_low_control, X=X, y=y, alpha=LOWER_ALPHA / 2, group=0,
                                     param_grid=param_grid)
                time_tune0_low = time.time() - start_tune0
                quantile_model_low_control = GradientBoostingRegressor(loss='quantile', alpha=LOWER_ALPHA / 2,
                                                                       **best_params_)

                hyperparameters[f'{LOWER_ALPHA / 2}_0'] = best_params_
            quantile_model_low_control.fit(X[g == 0, :], y[g == 0])
            time_control = time.time() - start

            time_pi[alpha][f'QR_Agnostic'] = time_treat + time_control
            print('Finished QR Tuning')

            ## 3 QR # Model Agnostic + Two Model ##
            start = time.time()
            if TUNE_PI == True:

                quantile_high_two_model_gbt = TwoModelRegressor(
                    control_group_model=GradientBoostingRegressor(loss='quantile', alpha=UPPER_ALPHA / 2,
                                                                  **hyperparameters[f'{UPPER_ALPHA / 2}_0']),
                    treatment_group_model=GradientBoostingRegressor(loss='quantile', alpha=UPPER_ALPHA / 2,
                                                                    **hyperparameters[f'{UPPER_ALPHA / 2}_1'])
                )
            else:
                quantile_high_two_model_gbt = TwoModelRegressor(
                    control_group_model=GradientBoostingRegressor(loss='quantile', alpha=UPPER_ALPHA / 2),
                    treatment_group_model=GradientBoostingRegressor(loss='quantile', alpha=UPPER_ALPHA / 2)
                )
            # %%
            quantile_high_two_model_gbt.fit(X=X[c == 1, :], y=y[c == 1], g=g[c == 1])
            high_time = time.time() - start

            if TUNE_PI == True:
                high_time += time_tune0_high + time_tune1_high
            # %%
            train_pred_high_treat = quantile_high_two_model_gbt.treatment_group_model.predict(X)
            test_pred_high_treat = quantile_high_two_model_gbt.treatment_group_model.predict(X_test)
            train_pred_high_control = quantile_high_two_model_gbt.control_group_model.predict(X)
            test_pred_high_control = quantile_high_two_model_gbt.control_group_model.predict(X_test)
            # %%

            # lower quantile
            start = time.time()
            if TUNE_PI == True:
                quantile_low_two_model_gbt = TwoModelRegressor(
                    control_group_model=GradientBoostingRegressor(loss='quantile', alpha=LOWER_ALPHA / 2,
                                                                  **hyperparameters[f'{LOWER_ALPHA / 2}_0']),
                    treatment_group_model=GradientBoostingRegressor(loss='quantile', alpha=LOWER_ALPHA / 2,
                                                                    **hyperparameters[f'{LOWER_ALPHA / 2}_1'])
                )
            else:
                quantile_low_two_model_gbt = TwoModelRegressor(
                    control_group_model=GradientBoostingRegressor(loss='quantile', alpha=LOWER_ALPHA / 2),
                    treatment_group_model=GradientBoostingRegressor(loss='quantile', alpha=LOWER_ALPHA / 2)
                )
            # %%
            quantile_low_two_model_gbt.fit(X=X[c == 1, :], y=y[c == 1], g=g[c == 1])
            low_time = time.time() - start
            if TUNE_PI == True:
                low_time += time_tune0_low + time_tune1_low
            # %%
            train_pred_low_treat = quantile_low_two_model_gbt.treatment_group_model.predict(X)
            test_pred_low_treat = quantile_low_two_model_gbt.treatment_group_model.predict(X_test)
            train_pred_low_control = quantile_low_two_model_gbt.control_group_model.predict(X)
            test_pred_low_control = quantile_low_two_model_gbt.control_group_model.predict(X_test)
            # %%
            PI_ITE_low_train = train_pred_low_treat - train_pred_high_control
            PI_ITE_low_test = test_pred_low_treat - test_pred_high_control
            PI_ITE_up_train = train_pred_high_treat - train_pred_low_control
            PI_ITE_up_test = test_pred_high_treat - test_pred_low_control

            prediction_intervals_train[alpha]['Agnostic_QR_two-model']['quantile_model'] = {}
            prediction_intervals_test[alpha]['Agnostic_QR_two-model']['quantile_model'] = {}
            prediction_intervals_train[alpha]['Agnostic_QR_two-model']['quantile_model']['pred_low'] = PI_ITE_low_train
            prediction_intervals_train[alpha]['Agnostic_QR_two-model']['quantile_model']['pred_high'] = PI_ITE_up_train
            prediction_intervals_test[alpha]['Agnostic_QR_two-model']['quantile_model']['pred_low'] = PI_ITE_low_test
            prediction_intervals_test[alpha]['Agnostic_QR_two-model']['quantile_model']['pred_high'] = PI_ITE_up_test
            time_pi[alpha][f'Agnostic_QR_two-model'] = high_time + low_time
            print('Finished QR two model')

        #### CQR with Package ####

        if CQR == True:
            n_estimators = 1000  # the number of trees in the forest

            # the minimum number of samples required to be at a leaf node # (default skgarden's parameter)
            min_samples_leaf = 1

            # the number of features to consider when looking for the best split # (default skgarden's parameter)
            max_features = X.shape[1]

            # target quantile levels
            quantiles_forest = [LOWER_ALPHA * 100, UPPER_ALPHA * 100]

            # use cross-validation to tune the quantile levels?
            cv_qforest = True

            # when tuning the two QRF quantile levels one may ask for a prediction band with smaller average coverage
            # to avoid too conservative estimation of the prediction band
            # This would be equal to coverage_factor*(quantiles[1] - quantiles[0])
            coverage_factor = 0.85

            # ratio of held-out data, used in cross-validation
            cv_test_ratio = 0.05

            # seed for splitting the data in cross-validation. Also used as the seed in quantile random forests function
            cv_random_state = 1

            # determines the lowest and highest quantile level parameters.
            # This is used when tuning the quanitle levels by cross-validation.
            # The smallest value is equal to quantiles[0] - range_vals.
            # Similarly, the largest value is equal to quantiles[1] + range_vals.
            cv_range_vals = 30

            # sweep over a grid of length num_vals when tuning QRF's quantile parameters
            cv_num_vals = 10

            X_train_val = X
            y_train_val = np.hstack([y])
            g_train_val = np.hstack([g])

            # compute input dimensions
            n_train = X_train_val.shape[0]
            in_shape = X_train_val.shape[1]

            # divide the data into proper training set and calibration set
            idx = np.random.permutation(n_train)
            n_half = int(np.floor(n_train / 2))
            idx_train, idx_cal = idx[:n_half], idx[n_half:2 * n_half]

            from cqr import helper
            from nonconformist.nc import RegressorNc
            from nonconformist.nc import QuantileRegErrFunc

            # define the QRF's parameters
            params_qforest = dict()
            params_qforest["n_estimators"] = n_estimators
            params_qforest["min_samples_leaf"] = min_samples_leaf
            params_qforest["max_features"] = max_features
            params_qforest["CV"] = cv_qforest
            params_qforest["coverage_factor"] = coverage_factor
            params_qforest["test_ratio"] = cv_test_ratio
            params_qforest["random_state"] = cv_random_state
            params_qforest["range_vals"] = cv_range_vals
            params_qforest["num_vals"] = cv_num_vals

            #### CQR from Package ###

            # note that the results for CQR, trained only on the converted customers is not depicted here
            # I ran it seperately and merged the results since the individuals in the folds are the same

            for error_function in [QuantileRegErrFunc(), QuantileRegAsymmetricErrFunc()]:  # ,

                #### CQR with Two Model ####
                start = time.time()
                quantile_estimator_treat = helper.QuantileForestRegressorAdapter(model=None,
                                                                                 fit_params=None,
                                                                                 quantiles=quantiles_forest,
                                                                                 params=params_qforest)

                # define the CQR object
                nc = RegressorNc(quantile_estimator_treat, error_function)

                # %%

                # run CQR procedure
                # use idx to select only those who are in right group
                test_pred_low_treat, test_pred_high_treat, train_pred_low_treat, train_pred_high_treat = helper.run_icp(
                    nc, X_train_val, y_train_val,
                    X_test,
                    idx_train[g_train_val[idx_train] == 1],
                    idx_cal[g_train_val[idx_cal] == 1],
                    alpha / 2,
                    train=True)

                # %%

                # define QRF model
                quantile_estimator_control = helper.QuantileForestRegressorAdapter(model=None,
                                                                                   fit_params=None,
                                                                                   quantiles=quantiles_forest,
                                                                                   params=params_qforest)

                # define the CQR object
                nc = RegressorNc(quantile_estimator_control, error_function)

                # %%
                # run CQR procedure
                # use idx to select only those who are in right group
                test_pred_low_control, test_pred_high_control, train_pred_low_control, train_pred_high_control = helper.run_icp(
                    nc, X_train_val, y_train_val,
                    X_test,
                    idx_train[g_train_val[idx_train] == 0],
                    idx_cal[g_train_val[idx_cal] == 0],
                    alpha / 2,
                    train=True)
                time_cqr = time.time() - start

                # %%

                PI_ITE_low_test = test_pred_low_treat - test_pred_high_control
                PI_ITE_up_test = test_pred_high_treat - test_pred_low_control
                PI_ITE_low_train = train_pred_low_treat - train_pred_high_control
                PI_ITE_up_train = train_pred_high_treat - train_pred_low_control

                error_string = str(error_function)
                error_string = error_string[18:35]
                prediction_intervals_train[alpha][f'CQR_two-model_rf_{error_string}']['quantile_model'] = {}
                prediction_intervals_test[alpha][f'CQR_two-model_rf_{error_string}']['quantile_model'] = {}
                prediction_intervals_test[alpha][f'CQR_two-model_rf_{error_string}']['quantile_model'][
                    'pred_low'] = PI_ITE_low_test
                prediction_intervals_test[alpha][f'CQR_two-model_rf_{error_string}']['quantile_model'][
                    'pred_high'] = PI_ITE_up_test
                prediction_intervals_train[alpha][f'CQR_two-model_rf_{error_string}']['quantile_model'][
                    'pred_low'] = PI_ITE_low_train
                prediction_intervals_train[alpha][f'CQR_two-model_rf_{error_string}']['quantile_model'][
                    'pred_high'] = PI_ITE_up_train
                print(f'Finished CQR - Two Model RF for {error_string}')
                time_pi[alpha][f'CQR_two-model_rf_{error_string}'] = time_cqr

            #### Local Split Conformal ####

            SEED = 1
            # Control Group
            mean_estimator = RandomForestRegressor(n_estimators=n_estimators,
                                                   min_samples_leaf=min_samples_leaf,
                                                   max_features=max_features,
                                                   random_state=SEED)

            # define the MAD estimator as random forests (used to scale the absolute residuals)
            mad_estimator = RandomForestRegressor(n_estimators=n_estimators,
                                                  min_samples_leaf=min_samples_leaf,
                                                  max_features=max_features,
                                                  random_state=SEED)

            # define a conformal normalizer object that uses the two regression functions.
            # The nonconformity score is absolute residual error
            normalizer = RegressorNormalizer(mean_estimator,
                                             mad_estimator,
                                             AbsErrorErrFunc())

            # define the final local conformal object
            nc = RegressorNc(mean_estimator, AbsErrorErrFunc(), normalizer)

            # build the split local conformal object
            icp = IcpRegressor(nc)

            # fit the conditional mean and MAD models to proper training data
            icp.fit(X_train_val[idx_train[g_train_val[idx_train] == 0]],
                    y_train_val[idx_train[g_train_val[idx_train] == 0]])

            # compute the absolute residual error on calibration data
            icp.calibrate(X_train_val[idx_cal[g_train_val[idx_cal] == 0]],
                          y_train_val[idx_cal[g_train_val[idx_cal] == 0]])

            # produce predictions for the test set, with confidence equal to significance
            predictions = icp.predict(X_test, significance=alpha / 2)
            predictions_train = icp.predict(X, significance=alpha / 2)

            # extract the lower and upper bound of the prediction interval
            test_pred_low_control = predictions[:, 0]
            test_pred_high_control = predictions[:, 1]
            train_pred_low_control = predictions_train[:, 0]
            train_pred_high_control = predictions_train[:, 1]

            # Treatment Group
            mean_estimator = RandomForestRegressor(n_estimators=n_estimators,
                                                   min_samples_leaf=min_samples_leaf,
                                                   max_features=max_features,
                                                   random_state=SEED)

            # define the MAD estimator as random forests (used to scale the absolute residuals)
            mad_estimator = RandomForestRegressor(n_estimators=n_estimators,
                                                  min_samples_leaf=min_samples_leaf,
                                                  max_features=max_features,
                                                  random_state=SEED)

            # %%

            # define a conformal normalizer object that uses the two regression functions.
            # The nonconformity score is absolute residual error
            normalizer = RegressorNormalizer(mean_estimator,
                                             mad_estimator,
                                             AbsErrorErrFunc())

            # define the final local conformal object
            nc = RegressorNc(mean_estimator, AbsErrorErrFunc(), normalizer)

            # build the split local conformal object
            icp = IcpRegressor(nc)

            # %%

            # fit the conditional mean and MAD models to proper training data
            icp.fit(X_train_val[idx_train[g_train_val[idx_train] == 1]],
                    y_train_val[idx_train[g_train_val[idx_train] == 1]])

            # compute the absolute residual error on calibration data
            icp.calibrate(X_train_val[idx_cal[g_train_val[idx_cal] == 1]],
                          y_train_val[idx_cal[g_train_val[idx_cal] == 1]])

            # produce predictions for the test set, with confidence equal to significance
            predictions = icp.predict(X_test, significance=alpha / 2)
            predictions_train = icp.predict(X, significance=alpha / 2)

            # extract the lower and upper bound of the prediction interval
            test_pred_low_treat = predictions[:, 0]
            test_pred_high_treat = predictions[:, 1]
            train_pred_low_treat = predictions_train[:, 0]
            train_pred_high_treat = predictions_train[:, 1]

            PI_ITE_low_test = test_pred_low_treat - test_pred_high_control
            PI_ITE_up_test = test_pred_high_treat - test_pred_low_control
            PI_ITE_low_train = train_pred_low_treat - train_pred_high_control
            PI_ITE_up_train = train_pred_high_treat - train_pred_low_control

            prediction_intervals_train[alpha][f'Local-CP_two-model_rf']['quantile_model'] = {}
            prediction_intervals_test[alpha][f'Local-CP_two-model_rf']['quantile_model'] = {}
            prediction_intervals_test[alpha][f'Local-CP_two-model_rf']['quantile_model']['pred_low'] = PI_ITE_low_test
            prediction_intervals_test[alpha][f'Local-CP_two-model_rf']['quantile_model']['pred_high'] = PI_ITE_up_test
            prediction_intervals_train[alpha][f'Local-CP_two-model_rf']['quantile_model']['pred_low'] = PI_ITE_low_train
            prediction_intervals_train[alpha][f'Local-CP_two-model_rf']['quantile_model']['pred_high'] = PI_ITE_up_train
            time_pi[alpha][f'Local-CP_two-model_rf'] = time_cqr
            print(f'Finished Local-CP - Two Model RF.')

            import torch
            # Neural network parameters  (shared by conditional quantile neural network
            # and conditional mean neural network regression)
            # See AllQNet_RegressorAdapter and MSENet_RegressorAdapter in helper.py
            nn_learn_func = torch.optim.Adam
            epochs = 1000
            lr = 0.0005
            hidden_size = 64
            batch_size = 64
            dropout = 0.1
            wd = 1e-6

            #### Local NN ####

            # Treatment Group
            start = time.time()
            model_treat = helper.MSENet_RegressorAdapter(model=None,
                                                         fit_params=None,
                                                         in_shape=in_shape,
                                                         hidden_size=hidden_size,
                                                         learn_func=nn_learn_func,
                                                         epochs=epochs,
                                                         batch_size=batch_size,
                                                         dropout=dropout,
                                                         lr=lr,
                                                         wd=wd,
                                                         test_ratio=cv_test_ratio,
                                                         random_state=cv_random_state)
            nc = RegressorNc(model_treat)

            test_pred_low_treat, test_pred_high_treat, train_pred_low_treat, train_pred_high_treat = helper.run_icp(nc,
                                                                                                                    X_train_val.astype(
                                                                                                                        np.float32),
                                                                                                                    y_train_val.astype(
                                                                                                                        np.float32),
                                                                                                                    X_test.astype(
                                                                                                                        np.float32),
                                                                                                                    idx_train[
                                                                                                                        g_train_val[
                                                                                                                            idx_train] == 1],
                                                                                                                    idx_cal[
                                                                                                                        g_train_val[
                                                                                                                            idx_cal] == 1],
                                                                                                                    alpha / 2,
                                                                                                                    train=True)
            # Control Group
            model_control = helper.MSENet_RegressorAdapter(model=None,
                                                           fit_params=None,
                                                           in_shape=in_shape,
                                                           hidden_size=hidden_size,
                                                           learn_func=nn_learn_func,
                                                           epochs=epochs,
                                                           batch_size=batch_size,
                                                           dropout=dropout,
                                                           lr=lr,
                                                           wd=wd,
                                                           test_ratio=cv_test_ratio,
                                                           random_state=cv_random_state)
            nc = RegressorNc(model_control)

            test_pred_low_control, test_pred_high_control, train_pred_low_control, train_pred_high_control = helper.run_icp(
                nc, X_train_val.astype(np.float32),
                y_train_val.astype(np.float32),
                X_test.astype(np.float32),
                idx_train[g_train_val[idx_train] == 0],
                idx_cal[g_train_val[idx_cal] == 0],
                alpha / 2, train=True)

            time_nn = time.time() - start

            # %%

            PI_ITE_low_test = test_pred_low_treat - test_pred_high_control
            PI_ITE_up_test = test_pred_high_treat - test_pred_low_control
            PI_ITE_low_train = train_pred_low_treat - train_pred_high_control
            PI_ITE_up_train = train_pred_high_treat - train_pred_low_control

            prediction_intervals_train[alpha][f'CP_two-model_NN']['quantile_model'] = {}
            prediction_intervals_test[alpha][f'CP_two-model_NN']['quantile_model'] = {}
            prediction_intervals_test[alpha][f'CP_two-model_NN']['quantile_model']['pred_low'] = PI_ITE_low_test
            prediction_intervals_test[alpha][f'CP_two-model_NN']['quantile_model']['pred_high'] = PI_ITE_up_test
            prediction_intervals_train[alpha][f'CP_two-model_NN']['quantile_model']['pred_low'] = PI_ITE_low_train
            prediction_intervals_train[alpha][f'CP_two-model_NN']['quantile_model']['pred_high'] = PI_ITE_up_train
            time_pi[alpha][f'CP_two-model_NN'] = time_nn
            print(f'Finished CP - Two Model NN.')

            # #### Neural Net with Local CP ####

            beta_net = 1
            start = time.time()

            # Treatment Group
            normalizer_adapter = helper.MSENet_RegressorAdapter(model=None,
                                                                fit_params=None,
                                                                in_shape=in_shape,
                                                                hidden_size=hidden_size,
                                                                learn_func=nn_learn_func,
                                                                epochs=epochs,
                                                                batch_size=batch_size,
                                                                dropout=dropout,
                                                                lr=lr,
                                                                wd=wd,
                                                                test_ratio=cv_test_ratio,
                                                                random_state=cv_random_state)
            adapter = helper.MSENet_RegressorAdapter(model=None,
                                                     fit_params=None,
                                                     in_shape=in_shape,
                                                     hidden_size=hidden_size,
                                                     learn_func=nn_learn_func,
                                                     epochs=epochs,
                                                     batch_size=batch_size,
                                                     dropout=dropout,
                                                     lr=lr,
                                                     wd=wd,
                                                     test_ratio=cv_test_ratio,
                                                     random_state=cv_random_state)

            normalizer = RegressorNormalizer(adapter,
                                             normalizer_adapter,
                                             AbsErrorErrFunc())
            nc = RegressorNc(adapter, AbsErrorErrFunc(), normalizer, beta=beta_net)

            # %%

            test_pred_low_treat, test_pred_high_treat, train_pred_low_treat, train_pred_high_treat = helper.run_icp(nc,
                                                                                                                    X_train_val.astype(
                                                                                                                        np.float32),
                                                                                                                    y_train_val.astype(
                                                                                                                        np.float32),
                                                                                                                    X_test.astype(
                                                                                                                        np.float32),
                                                                                                                    idx_train[
                                                                                                                        g_train_val[
                                                                                                                            idx_train] == 1],
                                                                                                                    idx_cal[
                                                                                                                        g_train_val[
                                                                                                                            idx_cal] == 1],
                                                                                                                    alpha / 2,
                                                                                                                    train=True)

            # Control group
            normalizer_adapter = helper.MSENet_RegressorAdapter(model=None,
                                                                fit_params=None,
                                                                in_shape=in_shape,
                                                                hidden_size=hidden_size,
                                                                learn_func=nn_learn_func,
                                                                epochs=epochs,
                                                                batch_size=batch_size,
                                                                dropout=dropout,
                                                                lr=lr,
                                                                wd=wd,
                                                                test_ratio=cv_test_ratio,
                                                                random_state=cv_random_state)
            adapter = helper.MSENet_RegressorAdapter(model=None,
                                                     fit_params=None,
                                                     in_shape=in_shape,
                                                     hidden_size=hidden_size,
                                                     learn_func=nn_learn_func,
                                                     epochs=epochs,
                                                     batch_size=batch_size,
                                                     dropout=dropout,
                                                     lr=lr,
                                                     wd=wd,
                                                     test_ratio=cv_test_ratio,
                                                     random_state=cv_random_state)

            normalizer = RegressorNormalizer(adapter,
                                             normalizer_adapter,
                                             AbsErrorErrFunc())
            nc = RegressorNc(adapter, AbsErrorErrFunc(), normalizer, beta=beta_net)

            # %%

            test_pred_low_control, test_pred_high_control, train_pred_low_control, train_pred_high_control = helper.run_icp(
                nc, X_train_val.astype(np.float32),
                y_train_val.astype(np.float32),
                X_test.astype(np.float32),
                idx_train[g_train_val[idx_train] == 0],
                idx_cal[g_train_val[idx_cal] == 0],
                alpha / 2, train=True)

            time_nn = time.time() - start

            # %%

            PI_ITE_low_test = test_pred_low_treat - test_pred_high_control
            PI_ITE_up_test = test_pred_high_treat - test_pred_low_control
            PI_ITE_low_train = train_pred_low_treat - train_pred_high_control
            PI_ITE_up_train = train_pred_high_treat - train_pred_low_control

            prediction_intervals_train[alpha][f'Local-CP_two-model_NN']['quantile_model'] = {}
            prediction_intervals_test[alpha][f'Local-CP_two-model_NN']['quantile_model'] = {}
            prediction_intervals_test[alpha][f'Local-CP_two-model_NN']['quantile_model']['pred_low'] = PI_ITE_low_test
            prediction_intervals_train[alpha][f'Local-CP_two-model_NN']['quantile_model'][
                'pred_high'] = PI_ITE_low_train
            prediction_intervals_test[alpha][f'Local-CP_two-model_NN']['quantile_model']['pred_low'] = PI_ITE_low_test
            prediction_intervals_train[alpha][f'Local-CP_two-model_NN']['quantile_model']['pred_high'] = PI_ITE_up_train
            time_pi[alpha][f'Local-CP_two-model_NN'] = time_nn
            print(f'Finished Local-CP - Two Model NN.')

        #### Jackknife/MAPIE #####
        if Jackknife == True:
            print('Start Jackknife.')
            # MAPIE PARAMS
            Parameters = TypedDict("Parameters", {"method": str, "cv": Union[int, Subsample]})
            STRATEGIES = {
                # "prefit": Parameters(cv='prefit'),
                "naive": Parameters(method="naive"),
                # "jackknife": Params(method="base", cv=-1),
                # "jackknife_plus": Params(method="plus", cv=-1),
                # "jackknife_minmax": Params(method="minmax", cv=-1),
                # "cv": Params(method="base", cv=10),
                "cv_plus": Parameters(method="plus", cv=5),
                # "cv_minmax": Params(method="minmax", cv=10),
                # "jackknife_plus_ab": Parameters(method="plus", cv=Subsample(n_resamplings=50)),
                # "jackknife_minmax_ab": Params(method="minmax", cv=Subsample(n_resamplings=50))
            }

            # ## 1 TWO MODEL HURDLE ##

            # CONTROL GROUP
            test_pred_control = {}
            train_pred_control = {}
            y_pred_control_train = {}
            y_pred_control, y_pis = {}, {}
            time_control = {}
            for strategy, Parameters in STRATEGIES.items():

                try:
                    start = time.time()
                    mapie = MapieRegressor(
                        treatment_model_lib["two-model_hurdle_gbt"].control_group_model.value_regressor,
                        **Parameters)

                    mapie.fit(X[g == 0, :], y[g == 0])  # ?
                    time_control[strategy] = time.time() - start
                    y_pred_control[strategy], test_pred_control[strategy] = mapie.predict(X_test, alpha=alpha / 2)
                    y_pred_control_train[strategy], train_pred_control[strategy] = mapie.predict(X, alpha=alpha / 2)

                except:
                    print('There was an issue with Jackknife-  two model hurdle.')

            # TREATMENT GROUP
            test_pred_treat = {}
            y_pred_treat_train = {}
            train_pred_treat = {}
            y_pred_treat, y_pis = {}, {}
            time_treat = {}
            for strategy, Parameters in STRATEGIES.items():
                try:
                    start = time.time()
                    mapie = MapieRegressor(
                        treatment_model_lib["two-model_hurdle_gbt"].treatment_group_model.value_regressor,
                        **Parameters)
                    mapie.fit(X[g == 1, :], y[g == 1])
                    time_treat = time.time() - start
                    time_pi[alpha][f'MAPIE_two-model_hurdle_{strategy}'] = time_treat + time_control[strategy]
                    y_pred_treat[strategy], test_pred_treat[strategy] = mapie.predict(X_test, alpha=alpha / 2)
                    y_pred_treat_train[strategy], train_pred_treat[strategy] = mapie.predict(X, alpha=alpha / 2)

                    PI_ITE_low_test = test_pred_treat[strategy][:, 0].ravel() - test_pred_control[strategy][:,
                                                                                1].ravel()
                    PI_ITE_low_train = train_pred_treat[strategy][:, 0].ravel() - train_pred_control[strategy][:,
                                                                                  1].ravel()
                    PI_ITE_up_test = test_pred_treat[strategy][:, 1].ravel() - test_pred_control[strategy][:, 0].ravel()
                    PI_ITE_up_train = train_pred_treat[strategy][:, 1].ravel() - train_pred_control[strategy][:,
                                                                                 0].ravel()

                    prediction_intervals_train[alpha][f'MAPIE_two-model_hurdle_{strategy}']['quantile_model'] = {}
                    prediction_intervals_test[alpha][f'MAPIE_two-model_hurdle_{strategy}']['quantile_model'] = {}
                    prediction_intervals_train[alpha][f'MAPIE_two-model_hurdle_{strategy}']['quantile_model'][
                        'pred_low'] = PI_ITE_low_train
                    prediction_intervals_train[alpha][f'MAPIE_two-model_hurdle_{strategy}']['quantile_model'][
                        'pred_high'] = PI_ITE_up_train
                    prediction_intervals_test[alpha][f'MAPIE_two-model_hurdle_{strategy}']['quantile_model'][
                        'pred_low'] = PI_ITE_low_test
                    prediction_intervals_test[alpha][f'MAPIE_two-model_hurdle_{strategy}']['quantile_model'][
                        'pred_high'] = PI_ITE_up_test


                except:
                    print(f'There was an issue with Jackknife-  two model hurdle for {strategy}.')

            print('Finished Jackknife + two model hurdle.')

            ## 2 TWO MODEL NO HURDLE##

            test_pred_control, train_pred_control, y_pred_control_train, y_pred_control, y_pis = {}, {}, {}, {}, {}

            time_control = {}
            # CONTROL GROUP
            for strategy, Parameters in STRATEGIES.items():
                try:
                    start = time.time()
                    mapie = MapieRegressor(two_model_outcome_gbt.control_group_model,
                                           **Parameters)
                    mapie.fit(X[g == 0, :], y[g == 0])  # ?
                    time_control[strategy] = time.time() - start
                    y_pred_control[strategy], test_pred_control[strategy] = mapie.predict(X_test, alpha=alpha / 2)
                    y_pred_control_train[strategy], train_pred_control[strategy] = mapie.predict(X, alpha=alpha / 2)

                except:
                    print(f'There was an issue with Jackknife - two model, no hurdle for {strategy}.')

                # TREATMENT GROUP
                test_pred_treat = {}
                y_pred_treat_train = {}
                train_pred_treat = {}
                y_pred_treat, y_pis = {}, {}
                # %%
            for strategy, Parameters in STRATEGIES.items():
                try:
                    start = time.time()
                    mapie = MapieRegressor(two_model_outcome_gbt.treatment_group_model,
                                           **Parameters)

                    mapie.fit(X[g == 1, :], y[g == 1])
                    time_treat = time.time() - start
                    time_pi[alpha][f'MAPIE_two-model_{strategy}'] = time_treat + time_control[strategy]
                    y_pred_treat[strategy], test_pred_treat[strategy] = mapie.predict(X_test, alpha=alpha / 2)
                    y_pred_treat_train[strategy], train_pred_treat[strategy] = mapie.predict(X, alpha=alpha / 2)
                    # %%
                    PI_ITE_low_test = test_pred_treat[strategy][:, 0].ravel() - test_pred_control[strategy][:,
                                                                                1].ravel()
                    PI_ITE_low_train = train_pred_treat[strategy][:, 0].ravel() - train_pred_control[strategy][:,
                                                                                  1].ravel()
                    PI_ITE_up_test = test_pred_treat[strategy][:, 1].ravel() - test_pred_control[strategy][:, 0].ravel()
                    PI_ITE_up_train = train_pred_treat[strategy][:, 1].ravel() - train_pred_control[strategy][:,
                                                                                 0].ravel()

                    prediction_intervals_train[alpha][f'MAPIE_two-model_{strategy}']['quantile_model'] = {}
                    prediction_intervals_test[alpha][f'MAPIE_two-model_{strategy}']['quantile_model'] = {}
                    prediction_intervals_train[alpha][f'MAPIE_two-model_{strategy}']['quantile_model'][
                        'pred_low'] = PI_ITE_low_train
                    prediction_intervals_train[alpha][f'MAPIE_two-model_{strategy}']['quantile_model'][
                        'pred_high'] = PI_ITE_up_train
                    prediction_intervals_test[alpha][f'MAPIE_two-model_{strategy}']['quantile_model'][
                        'pred_low'] = PI_ITE_low_test
                    prediction_intervals_test[alpha][f'MAPIE_two-model_{strategy}']['quantile_model'][
                        'pred_high'] = PI_ITE_up_test



                except:
                    print('There was an issue with Jackknife - two model, no hurdle.')

            print('Finished Jackknife, two model, no hurdle')


        else:
            print('No jackknife.')

    print(params)
    print(hyperparameters)

    ## Output formatting
    return ({"train": {"idx": split['train'],
                       "conversion": conversion_pred_train,
                       "treatment_conversion": treatment_conversion_train,
                       "treatment_basket_value": treatment_basketvalue_train,
                       "treatment_spending": treatment_pred_train,
                       "prediction_intervals": prediction_intervals_train,
                       "params": params,
                       "hyperparameters": hyperparameters,
                       "time_model": time_model,
                       "time_pi": time_pi,
                       },
             "test": {"idx": split['test'],
                      "conversion": conversion_pred_test,
                      "treatment_conversion": treatment_conversion_test,
                      "treatment_basket_value": treatment_basketvalue_test,
                      "treatment_spending": treatment_pred_test,
                      "prediction_intervals": prediction_intervals_test,
                      "time_model": time_model,
                      "time_pi": time_pi,
                      },
             })


#### Script
if __name__ == "__main__":

    # TUNING SETTINGS
    DEBUG = False  # if True: use smaller sample and only run one fold
    TUNE_CATE = False  # note: takes long time to run
    TUNE_PI = False

    # PI MODEL SPECIFICATION:
    qr = True
    Jackknife = True
    CQR = True
    xbcf = True

    today = date.today()

    # Load the data

    X = pd.read_csv("data/fashionB_clean_nonlinear.csv")

    # PARAMETERS
    SEED = 42
    N_SPLITS = 5
    np.random.seed(SEED)

    # Downsampling for debugging
    if DEBUG is True:
        X = X.sample(5000)
    print(X.shape)

    c = X.pop('converted')
    g = X.pop('TREATMENT')
    y = X.pop('checkoutAmount')
    tau_conversion = X.pop('TREATMENT_EFFECT_CONVERSION')
    tau_basket = X.pop('TREATMENT_EFFECT_BASKET')
    tau_response = X.pop('TREATMENT_EFFECT_RESPONSE')

    # Cross-validation folds stratified randomization by (outcome x treatment group)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # alternatively use for evaluation over more folds: RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=10, random_state=SEED)

    cg_groups = 2 * g + c  # Groups 0-4 depending on combinations [0,1]x[0,1]
    folds = list(splitter.split(X, cg_groups))
    print(folds)
    folds = get_train_validation_test_split(folds, val_set=False)
    print('Folds have been prepared for ' + str(len(folds)) + ' folds')

    # folds: 5 arrays each with train and test and their idx
    library_predictions = []


    def log_result(x):
        try:
            library_predictions.append(x)
        except Exception as e:
            library_predictions.append(str(e))


    if DEBUG is True:
        temp = predict_treatment_models(X=X, y=y, c=c, g=g, tau_conversion=tau_conversion, tau_basket=tau_basket,
                                        tau_response=tau_response, split=folds[0], fold_index=1, TUNE_PI=TUNE_PI,
                                        TUNE_CATE=TUNE_CATE,
                                        Jackknife=Jackknife, qr=qr, CQR=CQR, xbcf=xbcf)
        print(temp)
    else:
        pool = mp.Pool(N_SPLITS)

        for i, fold in enumerate(folds):
            pool.apply_async(predict_treatment_models,
                             args=(X, y, c, g, tau_conversion, tau_basket, tau_response, fold, i, TUNE_CATE, TUNE_PI,
                                   Jackknife, qr, CQR, xbcf),
                             callback=log_result)
        pool.close()
        pool.join()
        print("Cross-Validation complete.")

        np.save(f"results/run_thorugh_{today}", library_predictions, allow_pickle=True)

    print("Done!")
