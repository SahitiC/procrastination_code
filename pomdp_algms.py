"""
script contains functions for algorithms used to find optimal policy in POMDPs
The algorithms are numeric and based on dynamic programming. They follow a
grid-based method to discretise belief space (rbitrary grid size) associated
with a finite set of partially-observable states. The value function is
interpolated on-the-fly between the discrete belief states for the Bellman
update

Alternative method: pre-calculate a transition matrix between discrete
belief states by interpolation (YET TO IMPLEMENT)
"""

import math
import numpy as np
from scipy.interpolate import NearestNDInterpolator


def pO_ba(belief, action, observation, e_prob, t_prob):
    """
    probability of observation given the belief and action
    """
    return e_prob[action][:, observation].T @ (t_prob[action].T @ belief)


def get_next_belief(belief, action, observation, e_prob, t_prob):
    """
    finding the next belief state given belief, action and observation using
    bayes rule
    """
    return ((e_prob[action][:, observation] * (t_prob[action].T @ belief))
            / (pO_ba(belief, action, observation, e_prob, t_prob)))


def round_down(num, dx):
    """
    rounding down a number to precision of six decimal spaces
    """
    return math.floor(np.round(num / dx, 6)) * dx


def round_up(num, dx):
    """
    rounding up a number to precision of six decimal spaces
    """
    return math.ceil(np.round(num / dx, 6)) * dx


def interpolate1d(x, x_0, x_1, dx, func_0, func_1):
    """
    interpolate for a discrete 1D function
    """
    interpolated_func = 0.0
    interpolated_func = (((x-x_0)*func_1) + ((x_1-x)*func_0))/dx
    return interpolated_func


def interpolate_bilinear(x, x_0, x_1, dx, y, y_0, y_1, dy,
                         funcx0_y0, funcx1_y0, funcx0_y1, funcx1_y1):
    """
    interpolate for a discrete 2D function using 1D-interoplation
    (bilinear interpolation)
    """
    interpolate_y0 = interpolate1d(x, x_0, x_1, dx, funcx0_y0, funcx1_y0)
    interpolate_y1 = interpolate1d(x, x_0, x_1, dx, funcx0_y1, funcx1_y1)
    interpolated_func = interpolate1d(
        y, y_0, y_1, dy, interpolate_y0, interpolate_y1)
    return interpolated_func


def interpolate_triangle(x, x1, x2, x3, y, y1, y2, y3,
                         func1, func2, func3):
    """
    interpolate on a 2d triangle using barycentric coordinates (weights)
    this is necessary at the edge of belief space simplex
    three vertices: (x1, y1), (x2,y2), (x3,y3)
    point to interpolate on: (x, y)
    """
    # check if denominator is 0: happens when triangle is actually a line or
    # a point
    if ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)) == 0.0:
        return func1

    else:
        w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / \
            ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / \
            ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        w3 = 1.0 - w1 - w2
        return w1 * func1 + w2 * func2 + w3 * func3


def interpolate_value(belief, value, db):
    """
    interpolate value for an arbitrary belief given a discretised value
    function on a belief grid
    """
    mult = 1/db

    # if belief point lies exactly on grid, find exact value
    if (round_down(belief[0], db) == round_up(belief[0], db)
            and round_down(belief[1], db) == round_up(belief[1], db)):
        return value[int(belief[0] * mult), int(belief[1] * mult)]

    # if the outermost interpolation point is outside the grid,
    # barycentric interpolate on triangle
    elif round_up(belief[0], db) + round_up(belief[1], db) > 1.0:
        return interpolate_triangle(belief[0], round_down(belief[0], db),
                                    round_up(belief[0], db), round_down(
                                        belief[0], db),
                                    belief[1], round_down(belief[1], db),
                                    round_down(belief[1], db), round_up(
                                        belief[1], db),
                                    value[int(round_down(belief[0], db)*mult),
                                          int(round_down(belief[1], db)*mult)],
                                    value[int(round_up(belief[0], db)*mult),
                                          int(round_down(belief[1], db)*mult)],
                                    value[int(round_down(belief[0], db)*mult),
                                          int(round_up(belief[1], db)*mult)])

    # for all other points, bilinear interpolate
    else:
        return interpolate_bilinear(belief[0], round_down(belief[0], db),
                                    round_up(belief[0], db), db,
                                    belief[1], round_down(belief[1], db),
                                    round_up(belief[1], db), db,
                                    value[int(round_down(belief[0], db)*mult),
                                          int(round_down(belief[1], db)*mult)],
                                    value[int(round_up(belief[0], db)*mult),
                                          int(round_down(belief[1], db)*mult)],
                                    value[int(round_down(belief[0], db)*mult),
                                          int(round_up(belief[1], db)*mult)],
                                    value[int(round_up(belief[0], db)*mult),
                                          int(round_up(belief[1], db)*mult)])


def interpolate_policy(belief, policy, db):
    """
    interpolate policy on discrete belief grid based on nearest neighbor
    """
    mult = 1/db

    # for belief states at the edge (triangular coordinates)

    x = [(round_down(belief[0], db), round_down(belief[1], db)),
         (round_up(belief[0], db), round_down(belief[1], db)),
         (round_down(belief[0], db), round_up(belief[1], db))]
    y = [policy[int(round_down(belief[0], db)*mult),
                int(round_down(belief[1], db)*mult)],
         policy[int(round_up(belief[0], db)*mult),
                int(round_down(belief[1], db)*mult)],
         policy[int(round_down(belief[0], db)*mult),
                int(round_up(belief[1], db)*mult)]
         ]
    interp = NearestNDInterpolator(x, y)
    return interp(belief[0], belief[1])


def forward_runs_2D(belief, hidden_state, policy, db, states, observations,
                    e_prob, t_prob, t_prob_terminal):
    """
    sample trajectories given initial beleif, hidden state and policy
    """
    belief_trajectory = []
    state_trajectory = []
    action_trajectory = []
    observation_trajectory = []
    action = 10
    i_iter = 0

    # until action of submit is reached, loop
    while hidden_state != 3 and i_iter < 100:

        belief_trajectory.append(belief)
        state_trajectory.append(hidden_state)

        # pick based on policy over beliefs: interpolate!
        action = interpolate_policy(np.round(belief, 6), policy, db)
        action = int(np.round(action))
        action_trajectory.append(action)
        # action leads to a transition in hidden state (append terminal states)
        hidden_state = np.random.choice(np.append(states, 3),
                                        p=np.append(t_prob[action]
                                                    [hidden_state],
                                        t_prob_terminal[action]
                                                    [hidden_state]))
        # sample new observation after transition
        if hidden_state != 3:
            observation = np.random.choice(
                observations, p=e_prob[action][hidden_state])
            # belief update from observation
            belief = get_next_belief(
                belief, action, observation, e_prob, t_prob)
            observation_trajectory.append(observation)

        i_iter += 1

    return (belief_trajectory, state_trajectory, action_trajectory,
            observation_trajectory)


def get_optimal_policy_2D(states, actions, observations, e_prob, t_prob,
                          rewards, discount_factor,
                          db, max_iter, eps):
    """
    derive optimal policy given problem description for a case with 3 states
    and hence 2D belief space. If there is a terminal state, then include only
    the probability of transitioning to non-terminal states in the t_prob
    (and other matrices) the total transition probability to all states
    (including terminal states) adds up to 1
    """

    value = np.full((int(1/db)+1, int(1/db)+1), np.nan)
    policy = np.full((int(1/db)+1, int(1/db)+1), np.nan)
    b = np.arange(0, 1+db, db)

    # initialise value function (at 0.0) in the valid belief region
    for i_b1 in range(len(b)):

        for i_b2 in range(len(b)):

            if b[i_b1] + b[i_b2] <= 1.0:
                value[i_b1, i_b2] = 0.0

    i_iter = 0  # counter for number of value iterations
    value_new = np.copy(value)  # initialise array to store updated values
    diff_value = np.inf  # diff in value variable

    # stopping criteria: either greater than 100 iterations
    # or diff dips below eps
    while i_iter < max_iter and diff_value > eps:

        # loop through belief states
        for i_b1 in range(len(b)):

            for i_b2 in range(len(b)):

                if b[i_b1] + b[i_b2] <= 1.0:

                    belief = np.array([b[i_b1], b[i_b2], 1-b[i_b1]-b[i_b2]])

                    q = np.zeros((len(actions),))  # q-value for each action

                    # calculate q value for each action
                    for i_action, action in enumerate(actions):

                        future_value = 0.0

                        # loop through observations to calculate future value
                        # in non-terminal belief states
                        for i_observation, observation in enumerate(
                                observations):

                            # only consider observations (and transitions) that
                            # have non-zero probablity
                            if pO_ba(belief, action, observation,
                                     e_prob, t_prob) > 0.0:

                                # get b' given b, o, a
                                next_belief = get_next_belief(
                                    belief, action, observation, e_prob, t_prob
                                )

                                # make sure belief update <1.0
                                if sum(np.round(next_belief, 6)) > 1.0:
                                    print("encountered belief > 1.0")

                                # interpolate value at new belief and weight
                                # by probability of observing (P(O|b,a))
                                future_value = (future_value
                                                + pO_ba(belief, action,
                                                        observation, e_prob,
                                                        t_prob)
                                                * interpolate_value(
                                                    np.round(next_belief, 6),
                                                    value, db))

                                # make sure interpolation doesn't return
                                # non-valid values
                                if np.isnan(future_value):
                                    print("NaN encountered")

                        # Bellman update
                        q[action] = (belief.T @ rewards[action, :]
                                     + discount_factor * future_value)

                    # find max value and corresponding policy
                    value_new[i_b1, i_b2] = np.nanmax(q)
                    policy[i_b1, i_b2] = np.nanargmax(q)

        # find maximum difference in value between subsequent iterations
        diff_value = np.nanmax(np.abs(value_new-value))
        value = np.copy(value_new)
        i_iter += 1

    print(f"no. of iterations completed: {i_iter}")
    print(f"epsilon diff between subsequent values: {diff_value}")

    return policy, value
