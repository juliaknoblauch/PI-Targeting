# Eval helper
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, make_scorer, mean_pinball_loss
from copy import copy
import sys
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, ParameterGrid


sys.path.append(
    'C:/Users/julia/OneDrive - Humboldt-Universitaet zu Berlin, CMS/Desktop_alt/thesis/code/treatment-learn')
from treatlearn.policy import bayesian_targeting_policy
from treatlearn.evaluation import transformed_outcome_loss, expected_policy_profit

import matplotlib.pyplot as plt
import seaborn as sns
from myXBCF import myXBCF




# ##### Comparison on AUC

def calc_classification_error(prediction_dict, y_true, g):
    """
    Calculate the prediction error of the model predictions
    prediction_dict : dict
        Dictionary with the model predictions in the form model_name: array of predictions
    y_true : 1D array-like
        Observed outcomes
    g : 1D array-like
        Binary group indicator
    prob_treatment : array-like or int
        The group propensity for each observation. If None or int, the constant probability
        to be in binary treatment group 1.
    tau_true : 1D array-like
        Array of the true treatment effect. The true treatment effect
        is only known in simulations
    """
    output = {}

    for model_name, pred in prediction_dict.items():
        output[model_name] = {}
        output[model_name]["ROC-AUC"] = roc_auc_score(y_true=y_true[g], y_score=pred[g])
        output[model_name]["brier"] = mean_squared_error(y_pred=pred[g], y_true=y_true[g])

    return output


def calc_prediction_error(prediction_dict, y_true, g, time_dict=None, prob_treatment=None, tau_true=None):
    """
    Calculate the prediction error of the model predictions
    prediction_dict : dict
        Dictionary with the model predictions in the form model_name: array of predictions
    y_true : 1D array-like
        Observed outcomes
    g : 1D array-like
        Binary group indicator
    prob_treatment : array-like or int
        The group propensity for each observation. If None or int, the constant probability
        to be in binary treatment group 1.
    tau_true : 1D array-like
        Array of the true treatment effect. The true treatment effect
        is only known in simulations
    """
    output = {}

    if prob_treatment is None:
        prob_treatment = g.mean()

    for model_name, pred in prediction_dict.items():
        output[model_name] = {}

        pred = pred.clip(-200, 200)
        # print(model_name)

        output[model_name]["transformed_outcome_loss"] = transformed_outcome_loss(tau_pred=pred, y_true=y_true, g=g,
                                                                                  prob_treatment=prob_treatment)
        if tau_true is not None:
            output[model_name]["root_mean_squared_error"] = np.sqrt(mean_squared_error(y_pred=pred, y_true=tau_true))
            output[model_name]["mean_absolute_error"] = mean_absolute_error(y_pred=pred, y_true=tau_true)
        if time_dict is not None:
            try:
                output[model_name]['training_time'] = time_dict[model_name]
            except:
                output[model_name]['training_time'] = None
                # print(str(model_name)+' is skipped.')
    return output


def tune_threshold(treatment_dict, y_true, c_true, g, margin, contact_cost, offer_cost, prob_treatment=None):
    """
    Returns empirical cutoffs that give highest profit
    """
    # uses average treatment probability if not estimated
    if prob_treatment is None:
        prob_treatment = g.mean()

    # Threshold candiates [1, 0.975,...,0]
    # i.e we have 25 cutoff quantiles
    step_size = 25
    quantiles = np.array(range(1000, -1, -step_size)) / 1000

    threshold_dict = {}
    for treatment_model, treatment_pred in treatment_dict.items():
        quantile_candidates = np.quantile(treatment_pred, quantiles)
        # different quantiles depending on model

        best_profit = -np.inf
        best_threshold = None
        for threshold in quantile_candidates:
            # if value is bigger than treshold
            decision = (treatment_pred > threshold) * 1
            profit = expected_policy_profit(targeting_decision=decision, g=g,
                                            observed_profit=(y_true * margin - (
                                                        offer_cost * decision * c_true) - contact_cost),
                                            prob_treatment=prob_treatment)
            if profit > best_profit:
                best_threshold = threshold
                best_profit = profit

        threshold_dict[treatment_model] = best_threshold

    return threshold_dict


def tune_gamma(treatment_dict, pi_dict, conversion_dict, margin, contact_cost, offer_cost, y_true, c_true, g,
               prob_treatment=None):
    """ Emprical Tuning of Gamma for Regularization Policy (refered to as lambda in thesis)"""
    if prob_treatment is None:
        prob_treatment = g.mean()

    threshold_dict = {}

    for treatment_model, treatment_pred in treatment_dict.items():
        threshold_dict[treatment_model] = {}

        for pi_model, pi_preds in pi_dict[0.05].items():

            threshold_dict[treatment_model][pi_model] = {}
            try:
                pi_pred = pi_preds['quantile_model']['pred_low']

            except:
                print('something went wrong. Use tau_sd of 1')
                #tau_sd = 1
                pi_pred = 0

            start = 0
            stop = 1
            step = 0.01

            float_range_array = np.arange(start, stop, step)  #.round(2)

            for conversion_model, conversion_pred in conversion_dict.items():
                threshold_dict[treatment_model][pi_model][conversion_model] = {}

                best_profit = -np.inf
                best_threshold = None
                for threshold in float_range_array:

                    decision = ((
                                            treatment_pred - conversion_pred * offer_cost - contact_cost + threshold * pi_pred) > 0) * 1

                    profit = expected_policy_profit(targeting_decision=decision, g=g, observed_profit=(
                                y_true * margin - (offer_cost * decision * c_true)), prob_treatment=prob_treatment)
                    if profit > best_profit:
                        best_threshold = threshold
                        best_profit = profit

                threshold_dict[treatment_model][pi_model][conversion_model] = best_threshold

    return threshold_dict


def bayesian_uncertainty_targeting_policy(tau_pred, pi_pred, contact_cost, offer_accept_prob, offer_cost, gamma=0.1,
                                          value=None, ordering=False):
    """
    Applies regularization framework to make a targeting decision.
    The decision to target is made when the expected profit increase from targeting - a regularizing factor
    is strictly     larger than the expected cost of targeting

    tau_pred : array-like
      Treatment effect estimates given individual covariates

    pi_pred : array-like
      PI estimates around treatment effect
    contact_cost : float or array-like
      Static cost that realizes independent of outcome

    offer_accept_prob:array-like
        prediction for conversion probability

    offer_cost : float or array-like
      Cost that realizes when the offer is accepted

    gamma: float
        regularizing factor

    ordering: Boolean
        whether the policy should order the customers or target based on decision value, default: False

    value : float or array-like, default: None
      Value of the observations in cases where tau_pred is the
      change in acceptance probability (binary outcome ITE)

    return: decision: array-like
        if ordering is False: whether individuals are targeted by the policy (1) or not (0)
        if ordering is True: decision value of Regularization policy


    """
    # if value:
    #    tau_pred = tau_pred * value

    if ordering == False:
        decision = ((tau_pred - offer_accept_prob * offer_cost - contact_cost + gamma * pi_pred) > 0).astype('int')

    else:
        decision = tau_pred - offer_accept_prob * offer_cost - contact_cost + gamma * pi_pred

    return decision


def sharpe_policy(tau_pred, tau_sd, contact_cost, offer_accept_prob, offer_cost, R_B=0, ordering=False, value=None):
    """
    Applied the Sharpe policy to make a targeting decision.
    The policy calculates the value of the expected profit and dividies this by the width of the PI around
    the heterogeneous treatment effect


    tau_pred : array-like
      Treatment effect estimates given individual covariates

    tau_sd : array-like
      PI width around treatment effect

    offer_accept_prob:array-like
        prediction for conversion probability

    contact_cost : float or array-like
      Static cost that realizes independent of outcome

    offer_cost : float or array-like
      Cost that realizes when the offer is accepted

    R_B: float or array-like
        benchmark return, default: 0

    ordering: Boolean
        whether the policy should order the customers or target all, default: False

    value : float or array-like, default: None
      Value of the observations in cases where tau_pred is the
      change in acceptance probability (binary outcome ITE)

    return: decision: array-like
        if ordering is False: whether individuals are targeted by the policy (1) or not (0)
        if ordering is True: decision value of SR

    """

    R_CATE = ((tau_pred - offer_accept_prob * offer_cost - contact_cost) / (
                offer_accept_prob * offer_cost + contact_cost))

    SR = (R_CATE - R_B) / tau_sd

    if ordering == False:
        decision = ((SR > 0).astype('int'))

    else:
        decision = SR
    return decision

#%%

def calc_sharpe_policy_error(treatment_dict, pi_dict, conversion_dict, margin, contact_cost, offer_cost, ordering=False,
                             calc_error=False, y=None, g=None, tau_true=None, prob_treatment=None, policy_dict=None,
                             num_customers=None):
    """
    Calculates the Sharpe policy with RSME
    """

    policy = {}
    error = {}
    policy = {}
    TOL = {}
    RSME = {}
    Ratio_test = {}

    # Calculate targeting threshold according to expected value
    #try:
    for treatment_model, treatment_pred in treatment_dict.items():

        for pi_model, pi_preds in pi_dict[0.32].items():

            PI_low = pi_preds['quantile_model']['pred_low']
            PI_high = pi_preds['quantile_model']['pred_high']
            tau_sd = abs((PI_high - PI_low) / 2)

            #except:
            #    print('something went wrong. Use tau_sd of 1')
            #    tau_sd = 1

            for conversion_model, conversion_pred in conversion_dict.items():

                policy[str(treatment_model) + "+" + str(conversion_model) + "+" + str(
                    pi_model) + "+" + "sharpe" + '+' + str(num_customers)] = policy_dict[
                    str(treatment_model) + "+" + str(conversion_model) + "+" + str(
                        pi_model) + "+" + "sharpe" + '+' + str(num_customers)]

                if calc_error == True:
                    idx = policy[str(treatment_model) + "+" + str(conversion_model) + "+" + str(
                        pi_model) + "+" + "sharpe" + '+' + str(num_customers)] == 1
                    if idx.sum() != 0:
                        Ratio_test[str(treatment_model) + "+" + str(conversion_model) + "+" + str(
                            pi_model) + "+" + "sharpe" + '+' + str(num_customers)] = idx.mean()
                        TOL[str(treatment_model) + "+" + str(conversion_model) + "+" + str(
                            pi_model) + "+" + "sharpe" + '+' + str(num_customers)] = transformed_outcome_loss(
                            tau_pred=treatment_pred[[idx]], y_true=y[[idx]], g=g[[idx]], prob_treatment=prob_treatment)
                        RSME[str(treatment_model) + "+" + str(conversion_model) + "+" + str(
                            pi_model) + "+" + "sharpe" + '+' + str(num_customers)] = np.sqrt(
                            mean_squared_error(y_pred=treatment_pred[[idx]], y_true=tau_true[[idx]]))

    if calc_error == False:
        return policy
    else:
        return policy, {"TOL": TOL, "RSME": RSME, "Ratio_test": Ratio_test}


def calc_bayesian_policy(treatment_dict, conversion_dict, margin, contact_cost, offer_cost, calc_error=False,
                         y=None, g=None, tau_true=None, prob_treatment=None):
    """
    Bayesian/Analytical Policy Adapted to return the RSME among tageted custoners
    """
    if prob_treatment is None:
        prob_treatment = g.mean()
    policy = {}
    TOL = {}
    RSME = {}
    Ratio_test = {}
    pi_model = "None"
    # Calculate targeting threshold according to expected value
    for treatment_model, treatment_pred in treatment_dict.items():

        for conversion_model, conversion_pred in conversion_dict.items():

            policy["Bayesian+" + str(treatment_model) + "+" + str(conversion_model)] = bayesian_targeting_policy(
                tau_pred=treatment_pred * margin,
                offer_accept_prob=conversion_pred,
                contact_cost=contact_cost, offer_cost=offer_cost
            )
            if calc_error == True:
                idx = policy["Bayesian" + "+" + str(treatment_model) + "+" + str(conversion_model)] == 1
                Ratio_test["Bayesian" + "+" + str(treatment_model) + "+" + str(conversion_model)] = idx.mean()
                TOL["Bayesian" + "+" + str(treatment_model) + "+" + str(conversion_model)] = transformed_outcome_loss(
                    tau_pred=treatment_pred[[idx]], y_true=y[[idx]], g=g[[idx]], prob_treatment=prob_treatment)
                RSME["Bayesian" + "+" + str(treatment_model) + "+" + str(conversion_model)] = np.sqrt(
                    mean_squared_error(y_pred=treatment_pred[[idx]], y_true=tau_true[[idx]]))

    if calc_error == False:
        return policy
    else:
        return policy, {"TOL": TOL, "RSME": RSME, "Ratio_test": Ratio_test}


def calc_bayesian_uncertainty_policy(treatment_dict, pi_dict, conversion_dict, margin, contact_cost, offer_cost, gamma,
                                     tail='left', ordering=False,
                                     calc_error=False, y=None, g=None, tau_true=None, prob_treatment=None,
                                     gamma_dict=None):
    """
    Regularization Policy with optional return of RSME among treated invidiuals

    :param treatment_dict: dictionary with CATE and conversion estimates
    :param pi_dict: dictionary with PI estimates, seperate for alphas
    :param conversion_dict: dict of conersion estimates
    :param margin:float, profit margin
    :param contact_cost: float, treatment-dependent cost
    :param offer_cost: float, response-dependent cost
    :param gamma: float, regularizing factor
    :param tail: string, option to change boundary, default: 'left'
    :param ordering: numpy array, should customer be ordered according to the Sharpe values? default: False
    :param calc_error: boolean, calculate RSME among targeted customer?, default = False
    :return: policy dict

    """

    policy = {}
    TOL = {}
    RSME = {}
    Ratio_test = {}

    # Calculate targeting threshold according to expected value
    for treatment_model, treatment_pred in treatment_dict.items():

        for alpha in pi_dict:
            if alpha >= 0.15:
                continue

            for pi_model, pi_preds in pi_dict[alpha].items():

                try:
                    if tail == 'left':
                        pi_pred = pi_preds['quantile_model']['pred_low']

                    elif tail == 'right':
                        pi_pred = pi_preds['quantile_model']['pred_high']
                    #tau_sd = PI_high - PI_low


                except:
                    print('something went wrong. Use tau_sd of 1')

                for conversion_model, conversion_pred in conversion_dict.items():  #"Bayesian+"

                    policy["Regularization" + "+" + str(treatment_model) + "+" + str(conversion_model) + "+" + str(
                        pi_model) + "+" + str(gamma)] = bayesian_uncertainty_targeting_policy(
                        tau_pred=treatment_pred * margin,
                        pi_pred=pi_pred,
                        offer_accept_prob=conversion_pred,
                        contact_cost=contact_cost, offer_cost=offer_cost,
                        gamma=gamma,
                        ordering=ordering
                    )
                    if calc_error == True:
                        idx = policy["Regularization" + "+" + str(treatment_model) + "+" + str(
                            conversion_model) + "+" + str(pi_model) + "+" + str(gamma)] == 1
                        if idx.sum() != 0:
                            Ratio_test[
                                "Regularization" + "+" + str(treatment_model) + "+" + str(conversion_model) + "+" + str(
                                    pi_model) + "+" + str(gamma)] = idx.mean()
                            TOL["Regularization" + "+" + str(treatment_model) + "+" + str(conversion_model) + "+" + str(
                                pi_model) + "+" + str(gamma)] = transformed_outcome_loss(tau_pred=treatment_pred[[idx]],
                                                                                         y_true=y[[idx]], g=g[[idx]],
                                                                                         prob_treatment=prob_treatment)
                            RSME[
                                "Regularization" + "+" + str(treatment_model) + "+" + str(conversion_model) + "+" + str(
                                    pi_model) + "+" + str(gamma)] = np.sqrt(
                                mean_squared_error(y_pred=treatment_pred[[idx]], y_true=tau_true[[idx]]))

    if calc_error == False:
        return policy
    else:
        return policy, {"TOL": TOL, "RSME": RSME, "Ratio_test": Ratio_test}


def calc_bayesian_uncertainty_policy_fixed_gamma(treatment_dict, pi_dict, conversion_dict, margin, contact_cost,
                                                 offer_cost, gamma, tail='left', ordering=False):
    """
    This calculates the policy dict, i.e., whether an individual is treated (1) or not(0) based on the Sharpe metric
    :param treatment_dict: dictionary with CATE and conversion estimates
    :param pi_dict: dictionary with PI estimates, seperate for alphas
    :param conversion_dict: dict of conersion estimates
    :param margin:float, profit margin
    :param contact_cost: float, treatment-dependent cost
    :param offer_cost: float, response-dependent cost
    :param gamma: float, regularizing factor
    :param tail: string, option to change boundary, default: 'left'
    :param ordering: numpy array, should customer be ordered according to the Sharpe values? default: False
    :return: policy dict
    """

    policy = {}
    # Calculate targeting threshold according to expected value
    # try:
    for treatment_model, treatment_pred in treatment_dict.items():

        for alpha in pi_dict:
            if alpha >= 0.15:
                continue

            for pi_model, pi_preds in pi_dict[alpha].items():

                try:
                    if tail == 'left':
                        pi_pred = pi_preds['quantile_model']['pred_low']

                    elif tail == 'right':
                        pi_pred = pi_preds['quantile_model']['pred_high']

                except:
                    print('something went wrong. Use tau_sd of 1')


                for conversion_model, conversion_pred in conversion_dict.items():  # "Bayesian+"

                    # gamma = gamma_dict[treatment_model][pi_model][conversion_model]
                    # + "+" + str(gamma)
                    policy["Regularization" + "+" + str(treatment_model) + "+" + str(conversion_model) + "+" + str(
                        pi_model) + "+" + str(gamma)] = bayesian_uncertainty_targeting_policy(
                        tau_pred=treatment_pred * margin,
                        pi_pred=pi_pred,
                        offer_accept_prob=conversion_pred,
                        contact_cost=contact_cost, offer_cost=offer_cost,
                        gamma=gamma,
                        ordering=ordering
                    )

    return policy


def calc_bayesian_uncertainty_policy_fixed_gamma_error(treatment_dict, pi_dict, conversion_dict, margin, contact_cost,
                                                       offer_cost, gamma,
                                                       tail='left', ordering=False, calc_error=False, y=None, g=None,
                                                       tau_true=None, prob_treatment=None, policy_dict=None,
                                                       num_customers=None):
    """
    Regularization Policy with fixed gamma/lamda; note that if gamma=0 the policy is equivalent to the Analytical Policy
    """

    error = {}
    policy = {}
    TOL = {}
    RSME = {}
    Ratio_test = {}
    policy = {}
    # Calculate targeting threshold according to expected value
    #try:
    for treatment_model, treatment_pred in treatment_dict.items():

        for alpha in pi_dict:
            if alpha >= 0.15:
                continue

            for pi_model, pi_preds in pi_dict[alpha].items():
                #policy[alpha] = {}

                try:
                    if tail == 'left':
                        pi_pred = pi_preds['quantile_model']['pred_low']

                    elif tail == 'right':
                        pi_pred = pi_preds['quantile_model']['pred_high']
                    #tau_sd = PI_high - PI_low


                except:
                    print('something went wrong. Use tau_sd of 1')
                    #tau_sd = 1
                    PI_low = 0

                for conversion_model, conversion_pred in conversion_dict.items():  #"Bayesian+"

                    #gamma = gamma_dict[treatment_model][pi_model][conversion_model]
                    # + "+" + str(gamma)
                    policy["Regularization" + "+" + str(treatment_model) + "+" + str(conversion_model) + "+" + str(
                        pi_model) + "+" + str(gamma) + "+" + str(num_customers)] = policy_dict[
                        "Regularization" + "+" + str(treatment_model) + "+" + str(conversion_model) + "+" + str(
                            pi_model) + "+" + str(gamma) + "+" + str(num_customers)]

                    if calc_error == True:
                        idx = policy["Regularization" + "+" + str(treatment_model) + "+" + str(
                            conversion_model) + "+" + str(pi_model) + "+" + str(gamma) + '+' + str(num_customers)] == 1
                        if idx.sum() != 0:
                            Ratio_test[
                                "Regularization" + "+" + str(treatment_model) + "+" + str(conversion_model) + "+" + str(
                                    pi_model) + "+" + str(gamma) + '+' + str(num_customers)] = idx.mean()
                            TOL["Regularization" + "+" + str(treatment_model) + "+" + str(conversion_model) + "+" + str(
                                pi_model) + "+" + str(gamma) + '+' + str(num_customers)] = transformed_outcome_loss(
                                tau_pred=treatment_pred[[idx]], y_true=y[[idx]], g=g[[idx]],
                                prob_treatment=prob_treatment)
                            RSME[
                                "Regularization" + "+" + str(treatment_model) + "+" + str(conversion_model) + "+" + str(
                                    pi_model) + "+" + str(gamma) + '+' + str(num_customers)] = np.sqrt(
                                mean_squared_error(y_pred=treatment_pred[[idx]], y_true=tau_true[[idx]]))

    if calc_error == False:
        return policy
    else:
        return policy, {"TOL": TOL, "RSME": RSME, "Ratio_test": Ratio_test}


def calc_sharpe_policy(treatment_dict, pi_dict, conversion_dict, margin, contact_cost, offer_cost, ordering=False):
    """
    This calculates the policy dict, i.e., whether an individual is treated (1) or not(0) based on the Sharpe metric
    (version without RSME)
    :param treatment_dict: dictionary with CATE and conversion estimates
    :param pi_dict: dictionary with PI estimates, seperate for alphas
    :param conversion_dict: dict of conersion estimates
   :param margin:float, profit margin
    :param contact_cost: float, treatment-dependent cost
    :param offer_cost: float, response-dependent cost
    :param ordering: numpy array, should customer be ordered according to the Sharpe values? default: False
    :return: policy dict
    """
    policy = {}

    for treatment_model, treatment_pred in treatment_dict.items():

        for pi_model, pi_preds in pi_dict[0.32].items():

            PI_low = pi_preds['quantile_model']['pred_low']
            PI_high = pi_preds['quantile_model']['pred_high']
            tau_sd = abs((PI_high - PI_low) / 2)


            for conversion_model, conversion_pred in conversion_dict.items():
                policy[str(treatment_model) + "+" + str(conversion_model) + "+" + str(
                    pi_model) + "+" + "sharpe"] = sharpe_policy(
                    tau_pred=treatment_pred * margin,
                    tau_sd=tau_sd,
                    offer_accept_prob=conversion_pred,
                    contact_cost=contact_cost, offer_cost=offer_cost,
                    ordering=ordering
                )

    return policy

def select_customers(policy_dict_order, num_customers):

    """ Ranks customer according to the decision value and assigns policy decision under cutoff consideration """
    policy = {}
    for key, decision_values in policy_dict_order.items():

        sort_index = np.flip(np.argsort(decision_values))

        pos_idx = decision_values[sort_index] >= 0

        # only select subset that has decision values

        decision = np.zeros(decision_values.shape[0])

        if (decision_values[sort_index] >= 0).all():

            customers_idx = sort_index[:num_customers]
            decision[customers_idx] = 1

        else:
            customers_idx = sort_index[:num_customers] * pos_idx[:num_customers]
            decision[customers_idx] = 1

        policy[str(key) + '+' + str(num_customers)] = {}
        policy[str(key) + '+' + str(num_customers)] = decision

    return policy


def scale_PIs(prediction_interval_dict, tau_true):

    """Scale the PIs according to ITEs """

    output = {}

    for alpha, prediction_intervals in prediction_interval_dict.items():
        if alpha == 0.32:
            continue


        output[alpha] = {}

        for models in prediction_intervals:

            output[alpha][models] = {}
            width = {}
            coverage = {}
            #step = 0.1

            float_range_array_low = np.arange(0, 0.2, 0.01).round(2)

            float_range_array_high = np.arange(0, 4, 0.1).round(2)

            itertools.product(float_range_array_low, float_range_array_high)
            #float_range_list = list(float_range_array)

            for scaling in itertools.product(float_range_array_low, float_range_array_high):

                coverage[scaling] = {}
                coverage[scaling] = coverage_fraction(tau_true,
                                                      scaling[0] * prediction_intervals[models]['quantile_model'][
                                                          'pred_low'],
                                                      scaling[1] * prediction_intervals[models]['quantile_model'][
                                                          'pred_high'])
                #print(coverage[scaling])
                if coverage[scaling] >= (1 - alpha):
                    width[scaling] = {}
                    width[scaling] = width_fraction(
                        scaling[0] * prediction_intervals[models]['quantile_model']['pred_low'],
                        scaling[1] * prediction_intervals[models]['quantile_model']['pred_high'])

            try:
                output[alpha][models]['best_scaling'] = min(width, key=width.get)
            except:
                print(f'coverage not reached for {models}')
                output[alpha][models]['best_scaling'] = max(coverage, key=coverage.get)
                print(max(coverage, key=coverage.get))
                print(coverage)

    return output


def scale_PIs_CATE(prediction_interval_dict, treatment_dict):
    """
    Scale the boundaries of the PIs with the CATE estimates
    :param prediction_interval_dict: dict, dictionary of PI estimates
    :param treatment_dict: dict, dictionary of CATE estimates
    :return: dict, best scaling factor per combination of CATE model,alpha, PI model
    """

    output = {}

    tune_dict = {}
    for treatment_model, treatment_pred in treatment_dict.items():

        output[treatment_model] = {}

        for alpha, prediction_intervals in prediction_interval_dict.items():
            if alpha == 0.32:
                continue


            output[treatment_model][alpha] = {}

            for models in prediction_intervals:

                output[treatment_model][alpha][models] = {}
                width = {}
                coverage = {}
                step = 0.1

                float_range_array_low = np.arange(0.01, 0.2, 0.01).round(2)

                float_range_array_high = np.arange(0.01, 4, 0.1).round(2)

                itertools.product(float_range_array_low, float_range_array_high)
                #float_range_list = list(float_range_array)

                for scaling in itertools.product(float_range_array_low, float_range_array_high):
                    coverage[scaling] = {}

                    coverage[scaling] = coverage_fraction(treatment_pred,
                                                          scaling[0] * prediction_intervals[models]['quantile_model'][
                                                              'pred_low'],
                                                          scaling[1] * prediction_intervals[models]['quantile_model'][
                                                              'pred_high'])
                    if coverage[scaling] >= (1 - alpha):
                        width[scaling] = {}
                        width[scaling] = width_fraction(
                            scaling[0] * prediction_intervals[models]['quantile_model']['pred_low'],
                            scaling[1] * prediction_intervals[models]['quantile_model']['pred_high'])

                try:
                    output[treatment_model][alpha][models]['best_scaling'] = min(width, key=width.get)
                except:
                    print(f'coverage not reached for {models}')
                    output[treatment_model][alpha][models]['best_scaling'] = max(coverage, key=coverage.get)

    return output


def calc_naive_policy(treatment_dict):
    """
    Naive policy: expected profit if treat all or none
    """
    policy = {}
    # Calculate targeting threshold according to expected value
    for treatment_model, treatment_pred in treatment_dict.items():
        policy["Treat-all+"] = np.ones(treatment_pred.shape[0], dtype="int")
        policy["Treat-none+"] = np.zeros(treatment_pred.shape[0], dtype="int")

    return policy


def calc_threshold_policy(treatment_dict, threshold=0):
    """
    uses thresholds from empirical analysis above
    :param treatment_dict:
    :param threshold:
    :return:
    """
    policy = {}
    # Calculate targeting threshold according to expected value
    try:
        for treatment_model, treatment_pred in treatment_dict.items():
            policy["Threshold+" + str(treatment_model)] = (treatment_pred >= threshold[treatment_model]) * 1
    except:
        for treatment_model, treatment_pred in treatment_dict.items():
            policy["Threshold" + str(threshold) + "+" + str(treatment_model)] = (treatment_pred >= threshold) * 1

    return policy


def calc_policy_profit(policy_dict, y_true, c_true, g, margin, contact_cost, offer_cost, prob_treatment=None):
    if prob_treatment is None:
        prob_treatment = g.mean()

    profit = {key: expected_policy_profit(targeting_decision=decision, g=g,
                                          observed_profit=(y_true * margin - (
                                                      offer_cost * decision * c_true) - contact_cost),
                                          prob_treatment=prob_treatment).round(0)
              for key, decision in policy_dict.items()}
    ratio_treated = {key: decision.mean().round(2)
                     for key, decision in policy_dict.items()}

    return {"profit": profit, "ratio_treated": ratio_treated}


def coverage_fraction(y, y_low, y_high):
    """Calculates achieved PICP of y"""
    return np.mean(np.logical_and(y >= y_low, y <= y_high))


def width_fraction(y_low, y_high):
    """Calculates Width of Interval"""
    return np.mean(abs(y_high - y_low))


def calc_percentiles(prediction_interval_dict):
    """Calculates the percentile of widths achieved by the different models"""
    output = {}

    for alpha, prediction_intervals in prediction_interval_dict.items():
        if prediction_intervals == '':
            pass

        output[alpha] = {}

        for models in prediction_intervals:
            output[alpha][models] = {}

            # varibility of width
            widths_i = abs(prediction_intervals[models]['quantile_model']['pred_high'] -
                           prediction_intervals[models]['quantile_model']['pred_low'])
            df_describe = pd.DataFrame(widths_i)
            output[alpha][models]['width_percentiles'] = df_describe.describe()

    return output


def calc_uncertainty_metrics(prediction_interval_dict, time_dict, tau_true):
    """
    Calculate the PICP, MPIW, QNMPIW of the  PI model predictions
    prediction_interval_dict: dict
        Dictinoary of prediction intervals with order prediction low, prediction high.
        STD is already tranformed to prediction intervals.
    time_dict : dict
        Dictionary with the model predictions in the form model_name: array of predictions
    y_true : 1D array-like
        ITE
    return:
        dict
    """
    output = {}

    for alpha, prediction_intervals in prediction_interval_dict.items():
        if prediction_intervals == '':
            pass

        output[alpha] = {}

        for models in prediction_intervals:

            output[alpha][models] = {}

            output[alpha][models]['PICP'] = coverage_fraction(tau_true, prediction_intervals[models]['quantile_model'][
                'pred_low'], prediction_intervals[models]['quantile_model']['pred_high'])
            mpiw = width_fraction(prediction_intervals[models]['quantile_model']['pred_low'],
                                  prediction_intervals[models]['quantile_model'][
                                      'pred_high'])

            # range =tau_true.max() - tau_true.min()
            range = tau_true.quantile(q=1 - alpha) - tau_true.quantile(q=alpha)

            output[alpha][models]['QNMPIW'] = mpiw / range

            # varibility of width
            widths_i = abs(prediction_intervals[models]['quantile_model']['pred_high'] -
                           prediction_intervals[models]['quantile_model']['pred_low'])

            output[alpha][models]['Std'] = np.std(widths_i)

            try:
                output[alpha][models]['training_time'] = time_dict[alpha][models] / 60
            except:
                output[alpha][models]['training_time'] = None
                print(str(models) + ' is skipped since no training time for PI available.')

    return output


def transformed_outcome_loss(tau_pred, y_true, g, prob_treatment=None):
    """
    Calculate a biased estimate of the mean squared error of individualized treatment effects

    tau_pred : array
      The predicted individualized treatment effects.
    y_true : array
      The observed individual outcome.
    g : array, {0,1}
      An indicator of the treatment group. Currently supports only two treatment groups, typically
      control (g=0) and treatment group (g=1).
    """
    if prob_treatment is None:
        prob_treatment = g.mean()

    # Transformed outcome
    y_trans = (g - prob_treatment) * y_true / (prob_treatment * (1 - prob_treatment))
    loss = np.mean(((y_trans - tau_pred) ** 2))
    return loss


from itertools import product


def make_grid(iterables):
    """
    Create a list of tuples from the combinations of elements in several lists of different length

    Output
    ------
    list of tuples or list of dicts
      if iterables is a dictionary of lists, the output is a list of dictionaries with the same keys and
      values of each combination, e.g. {"A":[1,2], "B":[3]} -> [{"A":1, "B":3}, {"A":2, "B":3}]
    """
    if isinstance(iterables, dict):
        out = list(product(*iterables.values()))
        out = [dict(zip(iterables.keys(), x)) for x in out]
    else:
        out = list(product(*iterables))

    return out


def plot_cor_matrix(corr, mask=None, x_axis_labels=None):
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, ax=ax,
                mask=mask,
                # cosmetics
                annot=True, vmin=-1, vmax=1, center=0,
                cmap='coolwarm', linewidths=2, linecolor='black', cbar_kws={'orientation': 'vertical'},
                xticklabels=x_axis_labels, yticklabels=x_axis_labels)


def calculate_correlation_matrix_corr(prediction_interval_dict, tau_true, boundary='pred_low'):
    """
    Function that calculates the correlation within folds for ITE and PI boundaries

    :param prediction_interval_dict: dict, dict of prediction intervals
    :param tau_true: numpy array, ITE
    :param boundary: string, use upper or lower boundary, default: 'pred_low'
    :return: corr: dataframe
    """
    output = {}
    corr = {}

    for fold in range(len(prediction_interval_dict)):
        corr[fold] = {}
        output[fold] = {}
        idx = prediction_interval_dict[fold]['idx']
        tau_fold = tau_true[idx]

        for alpha, prediction_intervals in prediction_interval_dict[fold]['prediction_intervals'].items():
            if prediction_intervals == '':
                pass

            output[fold][alpha] = {}
            corr[fold][alpha] = {}

            for models in prediction_intervals:

                try:
                    output[fold][alpha][models] = {}
                    output[fold][alpha][models] = prediction_intervals[models]['quantile_model'][boundary]
                except:
                    print('some PI didnt work')

            df = pd.DataFrame.from_dict(output[fold][alpha], 'index').T
            df['tau_fold'] = tau_fold.values
            corr[fold][alpha] = df.corr()

    return corr


def calculate_correlation_matrix_corr_width(prediction_interval_dict, tau_true, CATE_means=None, CATE_model=None):
    """
    Function that calculates the correlation within folds for CATE estimate and PI widths

    :param prediction_interval_dict: dict, dict of prediction intervals
    :param tau_true: numpy array, ITE
    :param CATE_means: list, option to specific mean
    :param CATE_model: numpy array,
    :return: corr: dataframe,  deviations: dataframe
    """

    output = {}
    corr = {}
    deviations = {}

    for fold in range(len(prediction_interval_dict)):
        corr[fold] = {}
        deviations[fold] = {}
        output[fold] = {}
        idx = prediction_interval_dict[fold]['idx']
        tau_fold = tau_true[idx]

        if CATE_model is not None:
            CATEs = prediction_interval_dict[fold]['treatment_spending'][CATE_model]

            devs = abs(tau_fold - CATEs)

        else:
            print('Use mean of CATE Estimates.')
            CATEs = CATE_means[fold]
            devs = abs(tau_fold - CATEs)

        deviations[fold] = devs

        for alpha, prediction_intervals in prediction_interval_dict[fold]['prediction_intervals'].items():
            if prediction_intervals == '':
                pass

            output[fold][alpha] = {}
            corr[fold][alpha] = {}

            for models in prediction_intervals:
                output[fold][alpha][models] = {}
                PI_high = prediction_intervals[models]['quantile_model']['pred_high']
                PI_low = prediction_intervals[models]['quantile_model']['pred_low']
                width = abs((PI_high - PI_low)) / 2  # np.mean?
                output[fold][alpha][models] = width

            df = pd.DataFrame.from_dict(output[fold][alpha], 'index').T

            df['deviations'] = devs.reset_index(drop=True)

            corr[fold][alpha] = df.corr()

    return corr, deviations

    # Custom grid search function to optimize CATE models with extra argument treatment indicator g


def grid_search_cv(X, y, g, estimator, param_grid, folds, **kwargs):
    list_param_grid = list(ParameterGrid(param_grid))
    list_param_loss = []
    for param in list_param_grid:
        list_split_loss = []
        for split in folds:
            # Split the train and validation data
            _estimator = copy(estimator)
            X_test, y_test, g_test = [obj[split[1]] for obj in [X, y, g]]
            X_train, y_train, g_train = [obj[split[0]] for obj in [X, y, g]]
            _estimator.set_params(**param)
            _estimator.fit(X=X_train, y=y_train, g=g_train,
                           **{name: value[split[0]] for name, value in kwargs.items()})
            pred = _estimator.predict(X_test)
            tol = transformed_outcome_loss(pred, y_test, g_test)  # Minimize transformed outcome loss
            list_split_loss.append(tol)
        list_param_loss.append(np.mean(list_split_loss))

    return list_param_grid[list_param_loss.index(min(list_param_loss))]


def grid_search_cv_hurdle(X, y, g, estimator, param_grid_conversion, param_grid_regression, folds, **kwargs):
    list_param_grid = list(itertools.product(list(ParameterGrid(param_grid_conversion)),
                                             list(ParameterGrid(param_grid_regression))))
    list_param_loss = []
    for param in list_param_grid:
        list_split_loss = []
        for split in folds:
            # Split the train and validation data
            _estimator = copy(estimator)
            X_test, y_test, g_test = [obj[split[1]] for obj in [X, y, g]]
            X_train, y_train, g_train = [obj[split[0]] for obj in [X, y, g]]

            for model in [_estimator.treatment_group_model, _estimator.control_group_model]:
                model.conversion_classifier.set_params(**param[0])
                model.value_regressor.set_params(**param[1])

            _estimator.fit(X=X_train, y=y_train, g=g_train,
                           **{name: value[split[0]] for name, value in kwargs.items()})
            pred = _estimator.predict(X_test)
            tol = transformed_outcome_loss(pred, y_test, g_test)  # Minimize transformed outcome loss
            list_split_loss.append(tol)
        list_param_loss.append(np.mean(list_split_loss))
    return list_param_grid[list_param_loss.index(min(list_param_loss))]


# Custom grid search function to optimize CATE models with extra argument treatment indicator g
def grid_search_cv_xbcf(X, X_tr, y, g, estimator, param_grid, folds, **kwargs):
    """Grid search for XBCF that works but not used since too costly"""
    results = []
    # list_param_grid = list(ParameterGrid(param_grid))
    list_param_grid = make_grid(param_grid)
    print(list_param_grid)
    list_param_loss = []
    for i, param_set in enumerate(list_param_grid):
        list_split_loss = []
        print('parameter that are tried in this run')
        print(param_set)
        _param_set = copy(param_set)
        for split in folds:
            # Split the train and validation data
            cf = myXBCF(
                p_categorical_pr=param_set['p_categorical_pr'],
                p_categorical_trt=param_set['p_categorical_trt'],
                # model="Normal",
                parallel=False,  # because already parallelized?
                num_sweeps=param_set['num_sweeps'],
                burnin=param_set['burnin'],
                max_depth=250,
                num_trees_pr=param_set['num_trees_pr'],
                num_trees_trt=param_set['num_trees_trt'],
                num_cutpoints=param_set['num_cutpoints'],
                mtry_pr=param_set['mtry_pr'],
                mtry_trt=param_set['mtry_trt'],
                tau_pr=param_set['tau_pr'] * np.var(y) / param_set['num_trees_pr'],
                tau_trt=param_set['tau_trt'] * np.var(y) / param_set['num_trees_trt'],
                # no_split_penality="auto",
                alpha_pr=param_set['alpha_pr'],  # shrinkage (splitting probability)
                beta_pr=param_set['beta_pr'],  # shrinkage (tree depth)
                alpha_trt=param_set['alpha_trt'],  # shrinkage for treatment part
                beta_trt=param_set['beta_trt'],
            )
            _estimator = copy(cf)
            X_test, X_tr_test, y_test, g_test = [obj[split[1]] for obj in [X, X_tr, y, g]]
            X_train, X_tr_train, y_train, g_train = [obj[split[0]] for obj in [X, X_tr, y, g]]
            # _estimator.set_params(**param)
            print('estimator params')
            print(_estimator.params)
            _estimator.fit(x_t=X_tr_train, x=X_train, y=y_train, z=g_train.astype('int32'),
                           **{name: value[split[0]] for name, value in kwargs.items()})
            pred = _estimator.predict(X_test)
            tol = transformed_outcome_loss(pred, y_test,
                                           g_test.astype('int32'))  # Minimize transformed outcome loss
            list_split_loss.append(tol)
        list_param_loss.append(np.mean(list_split_loss))
        _param_set['tol'] = np.mean(list_split_loss)
        results.append(_param_set)

    return list_param_grid[list_param_loss.index(min(list_param_loss))], results



