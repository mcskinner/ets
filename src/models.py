import ets


# Basic models that you can get from the R forecast package.
SimpleETS = ets.Model([1], [[1]])
HoltsLinear = ets.Model([1, 1], [[1, 1], [0, 1]])
DampedTrendFromR = lambda cost_weight: ets.Model(
    [1, 'phi'],
    [[1, 'phi'], [0, 'phi']],
    var_bounds = {'phi': (0.8, 0.98)},
    soft_cost_weight = cost_weight)


# And then a wide variety of less traditional models.
# These were implemented after verifying that the basic results from above were sound.

# Simple damped model, no boundary constraints
UnconstrainedDampedTrend = lambda cost_weight: lambda cost_weight: ets.Model(
    [1, 'phi'],
    [[1, 'phi'], [0, 'phi']],
    soft_cost_weight = None)


# Also simple model, basic boundary constraints but nothing special on phi like R does.
DampedTrend = lambda cost_weight: ets.Model(
    [1, 'phi'],
    [[1, 'phi'], [0, 'phi']],
    soft_cost_weight = cost_weight)


# More interesting damped model, with different damping rates for different places.
TwoPhiAlpha = lambda cost_weight: ets.Model(
    [1, 'phi_w'],
    [[1, 'phi_F'], [0, 'phi_F']],
    soft_cost_weight = cost_weight)


# All the way loosened
TwoParamFullyLoosened = lambda cost_weight: ets.Model(
    [1, 1],
    [['a', 'b'], ['c', 'd']],
    start_state = [2.05, 3.85],
    start_params = [0.975, 0.25],
    var_init = {'a': 0.815, 'b': 0.195, 'c': 0, 'd': 1},
    soft_cost_weight = cost_weight)


# Hinted at by the results of the free parameter optimization
TwoPhiBeta = lambda cost_weight: ets.Model(
    [1, 1],
    [[1, 'phi_b'], [0, 'phi_d']],
    soft_cost_weight = cost_weight)


# Also hinted at as the best two-parameter form
FreeLevelUpdate = lambda cost_weight: ets.Model(
    [1, 1],
    [['a', 'b'], [0, 1]],
    start_state = [1.9, 3.97],
    start_params = [1.12, 0.05],
    var_init = {'a': 0.827, 'b': 0.16},
    soft_cost_weight = cost_weight)


# Probably the best formulation for expsmooth::bonds, in terms of AIC, requiring 3 parameters.
Triangular = lambda cost_weight: ets.Model(
    [1, 1],
    [['a', 'b'], [0, 'd']],
    start_state = [2, 4],
    start_params = [1, 0.1],
    var_init = {'a': 0.8, 'b': 0.2, 'd': 0.995},
    soft_cost_weight = cost_weight)
