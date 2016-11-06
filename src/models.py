import ets


# Basic models that you can get from the R forecast package.
SimpleETS = ets.Model([1], [[1]], param_vars = ['alpha'])
HoltsLinear = ets.Model([1, 1], [[1, 1], [0, 1]], param_vars = ['alpha', 'beta'])
DampedTrendFromR = lambda cost_weight: ets.Model(
    [1, 'phi'],
    [[1, 'phi'], [0, 'phi']],
    param_vars = ['alpha', 'beta'],
    var_bounds = {'phi': (0.8, 0.98)},
    soft_cost_weight = cost_weight)


# And then a wide variety of less traditional models.
# These were implemented after verifying that the basic results from above were sound.

# Simple damped model, no boundary constraints
UnconstrainedDampedTrend = lambda cost_weight: lambda cost_weight: ets.Model(
    [1, 'phi'],
    [[1, 'phi'], [0, 'phi']],
    param_vars = ['alpha', 'beta'],
    soft_cost_weight = None)


# Also simple model, basic boundary constraints but nothing special on phi like R does.
DampedTrend = lambda cost_weight: ets.Model(
    [1, 'phi'],
    [[1, 'phi'], [0, 'phi']],
    param_vars = ['alpha', 'beta'],
    soft_cost_weight = cost_weight)


# More interesting damped model, with different damping rates for different places.
TwoPhiAlpha = lambda cost_weight: ets.Model(
    [1, 'phi_w'],
    [[1, 'phi_F'], [0, 'phi_F']],
    param_vars = ['alpha', 'beta'],
    soft_cost_weight = cost_weight)


# All the way loosened
TwoParamFullyLoosened = lambda cost_weight: ets.Model(
    [1, 1],
    [['a', 'b'], ['c', 'd']],
    start_state = [2.05, 3.85],
    param_vars = ['alpha', 'beta'],
    var_init = {'alpha': 0.975, 'beta': 0.25, 'a': 0.815, 'b': 0.195, 'c': 0, 'd': 1},
    soft_cost_weight = cost_weight)


# Hinted at by the results of the free parameter optimization
TwoPhiBeta = lambda cost_weight: ets.Model(
    [1, 1],
    [[1, 'phi_b'], [0, 'phi_d']],
    param_vars = ['alpha', 'beta'],
    soft_cost_weight = cost_weight)


# Also hinted at as the best two-parameter form
FreeLevelUpdate = lambda cost_weight: ets.Model(
    [1, 1],
    [['a', 'b'], [0, 1]],
    start_state = [1.9, 3.97],
    param_vars = ['alpha', 'beta'],
    var_init = {'alpha': 1.12, 'beta': 0.05, 'a': 0.827, 'b': 0.16},
    soft_cost_weight = cost_weight)


# Probably the best formulation for expsmooth::bonds, in terms of AIC, requiring 3 parameters.
Triangular = lambda cost_weight: ets.Model(
    [1, 1],
    [['a', 'b'], [0, 'd']],
    start_state = [2, 4],
    param_vars = ['alpha', 'beta'],
    var_init = {'alpha': 1, 'beta': 0.1, 'a': 0.8, 'b': 0.2, 'd': 0.995},
    soft_cost_weight = cost_weight)


# A seasonal model for quarterly timeseries data (a frequency of 4).
QuarterlySeasonal = lambda cost_weight: ets.Model(
    [1, 1, 0, 0, 0],
    [
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0]
    ],
    start_state = [0, 0, 0, 0, 0],
    param_vars  = ['alpha', 0, 0, 0, 'gamma'],
    var_init = {'alpha': 0.99, 'gamma': 0.01},
    soft_cost_weight = cost_weight)


# The classic linear Holt Winters model for quarterly timeseries data (a frequency of 4).
QuarterlyHoltWinters = lambda cost_weight: ets.Model(
    [1, 1, 1, 0, 0, 0],
    [
        [1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0]
    ],
    start_state = [0, 0, 0, 0, 0, 0],
    param_vars  = ['alpha', 'beta', 0, 0, 0, 'gamma'],
    var_init = {'alpha': 0.99, 'beta': 0.01, 'gamma': 0.01},
    soft_cost_weight = cost_weight)


# A quarterly model that also maintains a sort of baseline steady state to revert to.
BaselineState = lambda cost_weight: ets.Model(
    [1, 0, 1, 0, 0, 0],
    [
        [1, 'reg', 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0]
    ],
    start_state = [0, 0, 0, 0, 0, 0],
    param_vars  = ['alpha', 'beta', 0, 0, 0, 'gamma'],
    var_init = {'alpha': 0.99, 'beta': 0.1, 'gamma': 0.01, 'reg': 0.0},
    var_bounds = {'reg': (-1, 0)},
    soft_cost_weight = cost_weight)
