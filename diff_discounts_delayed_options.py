
"""
script for mdp for choosing between two tasks each with delayed rewards.
The delay, rewards and costs can differ between the two options. In this case
too, reversals can be expected with different discount factors. Refer to
diff_discount_factors.py for assignment submission case where choice is between
shirking to get immediate rewards or working either with immediate rewards on
success or eward at the end. The case in this script is intended to be a
general version of this.

INCOMPLETE analysis
"""

import itertools
import mdp_algms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['lines.linewidth'] = 2

# %%

# construct separate reward and effort functions


def get_reward_functions(states, actions, reward_1, effort_1, delay_1,
                         reward_2, effort_2, delay_2, horizon):

    # for 2x2 states

    reward_func = []
    # for first state
    reward_func.append(np.zeros((len(actions[0]), horizon)))
    reward_func[0][0, delay_1] = reward_1  # action 1
    reward_func[0][1, delay_2] = reward_2  # action 2
    # for second state
    reward_func.append(np.zeros((len(actions[1]), horizon)))
    reward_func[1][0, delay_1] = reward_1  # action 1
    # for third state
    reward_func.append(np.zeros((len(actions[2]), horizon)))
    reward_func[2][0, delay_2] = reward_2  # action 1
    # for fourth state
    reward_func.append(np.zeros((len(actions[3]), horizon)))

    reward_func_last = [0., 0, 0, 0]

    cost_func = []
    # for first state
    cost_func.append(np.zeros((len(actions[0]), horizon)))
    cost_func[0][0, :] = effort_1  # action 1
    cost_func[0][1, :] = effort_2  # action 2
    # for second state
    cost_func.append(np.zeros((len(actions[1]), horizon)))
    cost_func[1][0, :] = effort_1  # action 1
    # for third state
    cost_func.append(np.zeros((len(actions[2]), horizon)))
    cost_func[2][0, :] = effort_2  # action 1
    # for fourth state
    cost_func.append(np.zeros((len(actions[3]), horizon)))

    cost_func_last = [0., 0, 0, 0]

    return (np.array(reward_func), np.array(reward_func_last),
            np.array(cost_func), np.array(cost_func_last))

# construct transition matrix


def get_transition_prob(states, efficacy):

    T = []

    # for 2x2 states:
    T.append(np.array([[1-efficacy, 0, efficacy, 0],
                       [1-efficacy, efficacy, 0, 0],
                       [1., 0, 0, 0]]))  # transitions for do_1, do_2, dont

    T.append(np.array([[0, 1-efficacy, 0, efficacy],
                       [0, 1., 0, 0]]))  # transitions for do_1, dont

    T.append(np.array([[0, 0, 1-efficacy, efficacy],
                       [0, 0, 1., 0]]))  # transitions for do_2, dont

    T.append(np.array([[0, 0, 0, 1.]]))  # transitions for dont

    return np.array(T)

# %%

# instantiate MDP


# states of markov chain
N_OPTIONS = 2  # NO. OF OPTIONS AVAILABLE
# STATES WITH 2 POSSIBLE STATES IN EACH OPTION
STATES = list(map(list, itertools.product([0, 1], repeat=N_OPTIONS)))

# ACTIONS AVAILABLE IN EACH STATE
ACTIONS = np.full(len(STATES), np.nan, dtype=object)
ACTIONS[0] = ['DO_1', 'DO_2', 'DONT']
ACTIONS[1] = ['DO_1', 'DONT']
ACTIONS[2] = ['DO_2', 'DONT']
ACTIONS[3] = ['DONT']

HORIZON = 10  # DEADLINE
DISCOUNT_FACTOR_REWARD = 0.9  # DISCOUNTING FACTOR FOR REWARDS
DISCOUNT_FACTOR_COST = 0.9  # DISCOUNTING FACTOR FOR COSTS
EFFICACY = 0.7  # SELF-EFFICACY (PROBABILITY OF PROGRESS ON WORKING)

# REWARDS AND CORRESPONDING DELAYS, EFFORTS FOR THE N_OPTIONS
DELAY_1 = 9
REWARD_1 = 8
EFFORT_1 = -1
DELAY_2 = 0
REWARD_2 = 0
EFFORT_2 = 0


# %%

reward_func, reward_func_last, cost_func, cost_func_last = (
    get_reward_functions(STATES, ACTIONS, REWARD_1, EFFORT_1, DELAY_1,
                         REWARD_2, EFFORT_2, DELAY_2, HORIZON)
)
T = get_transition_prob(STATES, EFFICACY)
V_opt_full, policy_opt_full, Q_values_full = (
    mdp_algms.find_optimal_policy_diff_discount_factors(
        STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST,
        reward_func, cost_func, reward_func_last, cost_func_last, T)
)
