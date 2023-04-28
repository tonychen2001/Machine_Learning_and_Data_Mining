import sys
import numpy as np
from genetic_algorithm import GeneticAlgorithm as ga
from plot_func import plot_3d
from statistics import NormalDist


###############################################################################
###############################################################################

# def normalize(arr):
#     """ This function normalizes a vector so the sum of its elements equal 1.

#     Args:
#     - arr (ndarray (shape: (k, 1))): A kX1 vector consisting k probability mixtures all in the range [0, 1].

#     Output:
#     - (ndarray (shape: (k, 1))): A kX1 vector consisting k normalized probability mixtures.
#     """

#     return arr / np.sum(arr)


def f(X):
    """ This function calculates the objective value.

    Args:
    - X (ndarray (shape: (3*k, 1))): A 3*kX1 vector consisting k sets of parameters for each gaussian component of the model (probability mixture, mean, standard deviation)
                                     Every 3rd element starting from index 0 is a probability mixture for component i corresponding to step i
                                     Every 3rd element starting from index 1 is a gaussian mean for component i corresponding to step i
                                     Every 3rd element starting from index 2 is a gaussian std. deviation for component i corresponding to step i

    Output:
    - float: objective value
    """

    n = 359  # number of observations of monthly VIX market prices

    num_bins = 48   # number of bins of histogram
    bin_width = (0.853 - (-0.486)) / num_bins    # width of each bin
    # the starting edges of each bin
    bins = np.array([-0.486, -0.45810417, -0.43020833, -0.4023125, -0.37441667, -0.34652083,
                     -0.318625, -0.29072917, -0.26283333, -0.2349375, -0.20704167, -0.17914583,
                     -0.15125, -0.12335417, -0.09545833, -0.0675625, -0.03966667, -0.01177083,
                     0.016125,  0.04402083,  0.07191667,  0.0998125,  0.12770833,  0.15560417,
                     0.1835,  0.21139583,  0.23929167,  0.2671875,  0.29508333,  0.32297917,
                     0.350875,  0.37877083,  0.40666667,  0.4345625,  0.46245833,  0.49035417,
                     0.51825,  0.54614583,  0.57404167,  0.6019375,  0.62983333,  0.65772917,
                     0.685625,  0.71352083,  0.74141667,  0.7693125,  0.79720833,  0.82510417])
    # the observed proportions of each bin
    observed_proportions = np.array(
        [1,  1,  2,  4,  1,  5,  3,  3,  6, 15, 14, 14, 16, 26, 19, 28, 23, 30,
         18, 18, 24, 18, 11,  7,  9,  8,  6,  3,  6,  5,  5,  3,  3,  0,  0,  0,
         0,  1,  1,  0,  1,  0,  0,  0,  0,  0,  0,  1]) / n

    lagrangeMul = X[len(X) - 1]  # extract lagrange multiplier
    theta = X[:len(X) - 1]      # extract all parameters of gaussian components

    dim = len(theta)

    assert dim % 3 == 0, f"The parameter length should be a multiple of 3 for every prob mixture, mean, standard deviation. Got {dim % 3}"

    k = int((dim) // 3)  # the number of regimes

    # extract all the probability mixtures from the paramaters
    probability_mixtures = theta[0::3]
    # extract all the means of the gaussian components from the paramaters
    gaussian_means = theta[1::3]
    # extract all the std deviations of the gaussian components from the paramaters
    gaussian_std_deviations = theta[2::3]

    assert len(
        probability_mixtures) == k, f"Number of prob mixtures should be {k}. Got {len(probability_mixtures)}"
    assert len(
        gaussian_means) == k, f"Number of prob mixtures should be {k}. Got {len(gaussian_means)}"
    assert len(
        gaussian_std_deviations) == k, f"Number of prob mixtures should be {k}. Got {len(gaussian_std_deviations)}"

    obj = 0

    for j in range(num_bins):
        if observed_proportions[j] == 0:
            continue

        # calculate p_j(theta)
        expected_proportion = 0
        for i in range(k):
            norm = NormalDist(
                mu=gaussian_means[i], sigma=gaussian_std_deviations[i])
            expected_proportion += probability_mixtures[i] * (
                norm.cdf(bins[j] + bin_width) - norm.cdf(bins[j]))
        expOverObs = expected_proportion / observed_proportions[j]
        obj += observed_proportions[j] * np.log(expOverObs)

    obj = -2 * n * obj

    return obj + (lagrangeMul * (np.absolute(np.sum(probability_mixtures) - 1)))


###############################################################################
###############################################################################


if __name__ == '__main__':

    # The number of regimes of the model (int from [1, 5])
    k = 5

    # Probability mixture belongs to [0, 1] ([0.05, 1] is satisfactory since we don't want a probability mixture to be 0 since the corresponding regime would essentially vanish)
    # Means of guassian distributions are unrestricted ([-1, 1] is satisfactory)
    # Standard deviation of guassian distributions belong to (0, inf) ([0.0001, 5] is satisfactory)
    thetaBound = np.array([[0.05, 1], [-1, 1], [0.0001, 5]]*k)
    # Langrage multiplier should be greater than 0 ([1000, 10000] was found to be good bounds during testing)
    lagrangeBound = np.array([1000, 10000])
    varbound = np.append(thetaBound, [lagrangeBound], axis=0)

    algorithm_param = {'max_num_iteration': 1000,
                       'population_size': 100,
                       'mutation_probability': 0.1,
                       'elit_ratio': 0.01,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.3,
                       'crossover_type': 'uniform',
                       'max_iteration_without_improv': None}

    model = ga(function=f, dimension=(3 * k) + 1,
               variable_type='real',
               variable_boundaries=varbound,
               algorithm_parameters=algorithm_param)

    model.run()

    # plot_3d(func=f, bounds=model.var_bound.flatten())
