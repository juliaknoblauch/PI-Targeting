from xbcausalforest import XBCF
import numpy as np

class myXBCF(XBCF):
    """ Monkeypatch s.t. same procedure for all models """

    def predict(self, X, *args, **kwargs):
        """Change predict such that it can be used with same predict structure as other CATE models"""

        CATE = super().predict(X, X, return_mean=True, return_muhat=False)

        return CATE

    # seperate model for std
    def predict_uncertainty(self, X, burnin_off=False):
        """
        Calculates the standard deviation of the posterior distribution
        Parameters
         ----------
        X : numpy array
            scaled covariates of observations
        burning_off: parameter, default: False
            if parameter is set to true it does not discard the burnin but uses it for prediction

        Returns
        -------
        std: np array, standard deviation
        """
        if burnin_off == True:
            posterior_distribution = super().predict(X, X, return_mean=False, return_muhat=False)
        else:
            posterior_distribution = super().predict(X, X, return_mean=False, return_muhat=False)[:,
                                     self.getParams()['burnin']:]

        std = np.std(posterior_distribution, axis=1)

        return std

    def get_params(self, deep=False):
        out = self.getParams()
        return out

    def predict_prediction_interval(self, X, LOWER_ALPHA, UPPER_ALPHA, burnin_off=False):
        """
        Calculates the PI from quantiles of posterior distribution
        Parameters
         ----------
        X : numpy array
            scaled covariates of observations
        LOWER_ALPHA, UPPER_APHA: float
            resulting quantile from miscoverage rate: lower usually alpha/2, upper usually: 1-alpha/2
        burning_off: parameter, default: False
            if parameter is set to true it does not discard the burnin but uses it for prediction

        Returns
        -------
        std: np array, standard deviation
        """
        if burnin_off == True:
            posterior_distribution = super().predict(X, X, return_mean=False, return_muhat=False)
        else:
            posterior_distribution = super().predict(X, X, return_mean=False, return_muhat=False)[:,
                                     self.getParams()['burnin']:]

        pred_low = np.quantile(posterior_distribution, LOWER_ALPHA, axis=1)
        pred_high = np.quantile(posterior_distribution, UPPER_ALPHA, axis=1)

        return pred_low, pred_high