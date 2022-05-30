# Customer Targeting with Prediction Intervals

## Replicability:

The code makes use of the more general helper library for uplift modeling: 

https://github.com/johaupt/treatment-learn

I want to thank Dr. Johannes Haupt for sharing the library!

For reproducing the environment use:

Create a conda environment with the environment.yml file with

conda env create --file environment.yml



## Abstract: 

The estimation of treatment effects on an individual level suffers from pronounced predictive uncertainty
as the true, individual treatment effect can never be determined. However, prediction
intervals around this unobservable effect can at least offer an estimate of the range it will be in.
Thus, this research examines the use of prediction intervals for heterogeneous treatment effects in
the targeting decision. I classify the methods for causal prediction interval estimation and find them
to be rarely used in marketing even though targeting policies base their decision on causal point
estimates. I evaluate selected methods that are distribution-free and model agnostic. Moreover, I
present different metrics to examine the intervals in a simulation and put forward a new metric that
allows for an easy comparison of the interval estimate’s quality. Among the evaluated methods, the
quantile-based approaches do not perform well for the imbalanced dataset and many approaches lead
to intervals that are not well calibrated to the true treatment effect. I further propose a calibration
mechanism that calibrates the intervals to the causal point estimates. Additionally, I propose
two new targeting policies based on financial metrics that utilize information from the intervals
differently: the Value at Risk policy utilizes the absolute value at the lower interval boundary and
the Sharpe policy utilizes the width of the interval. I find that the targeting policy based on the
Value at Risk does not reduce the uncertainty in the prediction or the profit, even though profit
could be increased if the intervals were calibrated better. The targeting based on the Sharpe policy
is successful in ranking the customers according to their predictive uncertainty. The findings show
the ability of prediction intervals to reflect the predictive uncertainty even in causal point estimates.
They further highlight that the goal of the targeting must be selected carefully, as a better predictive
ability not necessarily leads to higher profit.
