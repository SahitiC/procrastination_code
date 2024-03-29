"""
This is a script for an mdp when difficulty of the task is uncertain and the
only way to find out is to try the task. This then leads to probabilistic
transitions to one of N difficulty states which have different efforts
required to complete the task. These are subjective transitions in agent's
knowledge of th world rather than real transitions in the environment. Actions
and reward structure are like the simple case for assignment submission in
basic_discounting_model.py
"""

import seaborn as sns
import mdp_algms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams['text.usetex'] = False

# %%


def get_reward_functions(states, reward_pass, reward_fail, reward_shirk,
                         reward_completed, effort_try, effort_work,
                         effort_shirk):
    """
    construct reward function
    """
    # reward from actions within horizon
    reward_func = []
    # reward in start state
    reward_func.append([effort_try, reward_shirk + effort_shirk])

    for i in range(len(states)-2):
        # rewards in non-completed states
        reward_func.append([effort_work[i], reward_shirk + effort_shirk])

    reward_func.append([reward_completed])  # reward in completed state

    # reward from final evaluation
    reward_func_last = []
    reward_func_last.append(reward_fail)

    for i in range(len(states)-2):
        # rewards in non-completed states
        reward_func_last.append(reward_fail)

    reward_func_last.append(reward_pass)

    return (np.array(reward_func, dtype=object),
            np.array(reward_func_last, dtype=object))


def get_transition_prob(states, efficacy, difficulty_probability):
    """
    construct reward function
    """

    T = []
    T.append([np.array([0, difficulty_probability[0],
                        difficulty_probability[1], 0]),
              np.array([1, 0, 0, 0])])  # transitions for check, shirk in S=0
    T.append([np.array([0, 1-efficacy, 0, efficacy]),
              np.array([0, 1, 0, 0])])  # transitions for work, shirk
    T.append([np.array([0, 0, 1-efficacy, efficacy]),
              np.array([0, 0, 1, 0])])  # transitions for work, shirk
    T.append([np.array([0, 0, 0, 1])])  # transitions for completed

    return np.array(T, dtype=object)

# %%
# instantiating MDP


# states of markov chain
N_DIFFICULTY_STATES = 2  # number of states of difficulty
# all states including start and finished
STATES = np.arange(2 + N_DIFFICULTY_STATES)

ACTIONS = []
ACTIONS = [['work', 'shirk'] for i in range(len(STATES)-1)]
ACTIONS.append(['completed'])

HORIZON = 10  # deadline
DISCOUNT_FACTOR = 1.0  # discounting factor
# self-efficacy(probability of progress on working) in non-start/finished state
EFFICACY = 0.6
DIFFICULTY_PROBABILITY = [0.9, 0.1]  # probability for difficulty states

# utilities :
REWARD_PASS = 4.0
REWARD_FAIL = -4.0
REWARD_SHIRK = 0.5
EFFORT_TRY = -0.1  # effort to check
# effort to complete task from one of the difficulty states
EFFORT_WORK = [-0.1, -2.0]
EFFORT_SHIRK = -0
REWARD_COMPLETED = REWARD_SHIRK

# %%
# example policy

reward_func, reward_func_last = get_reward_functions(
    STATES, REWARD_PASS, REWARD_FAIL, REWARD_SHIRK,
    REWARD_COMPLETED, EFFORT_TRY, EFFORT_WORK, EFFORT_SHIRK
)
T = get_transition_prob(STATES, EFFICACY, DIFFICULTY_PROBABILITY)
V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(
    STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR,
    reward_func, reward_func_last, T
)

# plots of policies and values
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(STATES)))
lines = ['--', ':']
fig, axs = plt.subplots(figsize=(8, 6))
fig1, axs1 = plt.subplots(figsize=(5, 3))

for i_state, state in enumerate(STATES[:-1]):

    axs.plot(V_opt[i_state], color=colors[i_state], marker='o',
             linestyle='None', label=f'$V^*({i_state})$',)
    axs1.plot(policy_opt[i_state], color=colors[i_state],
              label=f'State {i_state}')

    for i_action, action in enumerate(ACTIONS[i_state]):

        axs.plot(Q_values[i_state][i_action, :],
                 color=colors[i_state], linestyle=lines[i_action])


handles, labels = axs.get_legend_handles_labels()
handles.append(axs.plot([], [], color='black', linestyle='--',
               label='$Q(a=$ check or work$)$'))
handles.append(axs.plot([], [], color='black',
               linestyle=':', label='$Q(a=$ shirk$)$'))

axs.legend()
axs.set_xlabel('timesteps')
axs1.legend()
axs1.set_xlabel('timesteps')
axs1.set_yticks([0, 1])
axs1.set_yticklabels(['WORK', 'SHIRK'])
axs1.set_ylabel('policy')

# %%
# solving for policies for a range of probabilities(of task being easy vs hard)

probabilities = np.linspace(0.0, 1.0, 5)
# optimal starting point for 4 reward regimes (change manually)
start_works = np.full((len(probabilities), N_DIFFICULTY_STATES+1), np.nan)

for i_prob, difficulty_probability in enumerate(probabilities):

    reward_func, reward_func_last = get_reward_functions(
        STATES, REWARD_PASS, REWARD_FAIL, REWARD_SHIRK,
        REWARD_COMPLETED, EFFORT_TRY, EFFORT_WORK, EFFORT_SHIRK
    )
    T = get_transition_prob(
        STATES, EFFICACY, [difficulty_probability, 1-difficulty_probability]
    )
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(
        STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, reward_func,
        reward_func_last, T
    )

    for i_state in range(N_DIFFICULTY_STATES+1):

        # find timepoints where it is optimal to work (when task not completed,
        # state=0)
        start_work = np.where(policy_opt[i_state, :] == 0)[0]

        if len(start_work) > 0:
            # first time to start working
            start_works[i_prob, i_state] = start_work[0]
            # print( policy_opt[0, :])

plt.figure(figsize=(8, 6))
colors = plt.cm.Blues(np.linspace(0.4, 0.9, N_DIFFICULTY_STATES+1))
for i_state in range(N_DIFFICULTY_STATES+1):

    plt.plot(probabilities, start_works[:, i_state],
             label=f'state = {i_state}', color=colors[i_state])

plt.xlabel('probability of task being easy')
plt.ylabel('time to start work/ check')
plt.legend()
plt.title(f'efficacy = {EFFICACY}')

# %%
# demonstration of procrastination due to discounting (no discounting vs
# discount factor < 1.0)

discount_factor = 0.9
efficacy = 0.6

reward_func, reward_func_last = get_reward_functions(
    STATES, REWARD_PASS, REWARD_FAIL, REWARD_SHIRK,
    REWARD_COMPLETED, EFFORT_TRY, EFFORT_WORK, EFFORT_SHIRK
)
T = get_transition_prob(STATES, efficacy, DIFFICULTY_PROBABILITY)
V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(
    STATES, ACTIONS, HORIZON, discount_factor, reward_func, reward_func_last, T
)

# plot trajectories
initial_state = 0
s, a, v = mdp_algms.forward_runs(
    policy_opt, V_opt, initial_state, HORIZON, STATES, T)

colors = ['tab:blue', 'brown']
# change action value of shirk in state=3 to get correct color
ac = np.where(s[:-1] == 3, 1, a)
colors_scatter = np.where(ac == 0, colors[0], colors[1])
colors_scatter = np.where((ac == 0) & (s[:-1] == 0), 'gold', colors_scatter)

plt.figure(figsize=(5, 4), dpi=100)
plt.plot(np.arange(0, HORIZON+1, 1),
         s, linewidth=2,
         linestyle='dashed',
         color='gray',
         label='state')
plt.scatter(np.arange(0, HORIZON, 1),
            s[:-1],
            marker='s', s=100,
            c=colors_scatter)
plt.xlabel('timesteps')
plt.yticks(STATES, labels=['i', 'e', 'h', 'm'])
plt.ylabel('state')
# plt.legend(frameon=False, loc='lower right')
sns.despine()

plt.savefig('writing/figures_thesis/vectors/planned_defections_discounted.svg',
            format='svg', dpi=300)

# %%
# shifting of checking time wrt finishing as function of difficulty probability

# solving for policies for a range of probabilities
N_runs = 1000
probabilities = np.linspace(0.0, 1.0, 5)
# optimal starting point for 4 reward regimes (change manually)
checking_times = np.zeros(len(probabilities))
# different columns for the different difficulty states
finishing_times = np.full((N_runs, len(probabilities), 2), np.nan)
initial_state = 0

for i_prob, difficulty_probability in enumerate(probabilities):

    reward_func, reward_func_last = get_reward_functions(
        STATES, REWARD_PASS, REWARD_FAIL, REWARD_SHIRK,
        REWARD_COMPLETED, EFFORT_TRY, EFFORT_WORK, EFFORT_SHIRK
    )
    T = get_transition_prob(
        STATES, EFFICACY, [difficulty_probability, 1-difficulty_probability]
    )
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(
        STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR,
        reward_func, reward_func_last, T
    )

    if len(np.where(policy_opt[0, :] == 0)[0]) > 0:
        checking_times[i_prob] = np.where(policy_opt[0, :] == 0)[0][0]

    for i in range(N_runs):

        s, a, v = mdp_algms.forward_runs(
            policy_opt, V_opt, initial_state, HORIZON, STATES, T)

        if 3 in s:
            if 1 in s:
                finishing_times[i, i_prob, 0] = np.where(s == 3)[0][0]
            elif 2 in s:
                finishing_times[i, i_prob, 1] = np.where(s == 3)[0][0]

plt.figure(figsize=(5, 4), dpi=100)
plt.plot(probabilities,
         checking_times,
         linewidth=2,
         color='tab:blue',
         label='check')
mean = np.nanmean(finishing_times[:, :, 0], axis=0)
std = np.nanstd(finishing_times[:, :, 0], axis=0)/np.sqrt(N_runs)
plt.plot(probabilities,
         mean,
         linewidth=2,
         color='brown',
         label='work in s=e')
plt.fill_between(probabilities,
                 mean-std,
                 mean+std,
                 alpha=0.3,
                 color='brown')
mean = np.nanmean(finishing_times[:, :, 1], axis=0)
std = np.nanstd(finishing_times[:, :, 1], axis=0)/np.sqrt(N_runs)
plt.plot(probabilities,
         mean,
         linewidth=2,
         color='tomato',
         label='work in s=h')
plt.fill_between(probabilities,
                 mean-std,
                 mean+std,
                 alpha=0.3,
                 color='tomato')
plt.xlabel('probability of task being easy')
plt.ylabel('avg time completion of action')
plt.legend(frameon=False)
sns.despine()

plt.savefig('writing/figures_thesis/vectors/planned_defections_checking_times.svg',
            format='svg', dpi=300)

# %%
# completion rates can be improved by improving reward
N_runs = 1000
initial_state = 2
completion_times = np.full((N_runs, 5), np.nan)
completion_rates = np.zeros((N_runs, 5))
rewards = np.array([0.0, 1.0, 2.0, 4.0, 6.0])
efficacy = 0.6

for i_r, reward_pass in enumerate(rewards):

    reward_fail = -4.0

    reward_func, reward_func_last = get_reward_functions(
        STATES, reward_pass, reward_fail, REWARD_SHIRK,
        REWARD_COMPLETED, EFFORT_TRY, EFFORT_WORK, EFFORT_SHIRK
    )
    T = get_transition_prob(
        STATES, efficacy, [difficulty_probability, 1-difficulty_probability]
    )
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(
        STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR,
        reward_func, reward_func_last, T
    )

    for i in range(N_runs):

        s, a, v = mdp_algms.forward_runs(
            policy_opt, V_opt, initial_state, HORIZON, STATES, T)
        # append completion time if task is completed
        if 3 in s:
            completion_rates[i, i_r] = 1.0
            completion_times[i, i_r] = np.where(s == 3)[0][0]

fig, axs = plt.subplots(figsize=(6, 4), dpi=100)

mean = np.nanmean(completion_times, axis=0)
std = np.nanstd(completion_times, axis=0)/np.sqrt(1000)
axs.plot(rewards,
         mean,
         linestyle='--',
         linewidth=2,
         marker='o', markersize=5,
         color='brown')

axs.fill_between(rewards,
                 mean-std,
                 mean+std,
                 alpha=0.3,
                 color='brown')

axs.set_xlabel('reward on completion')
axs.set_ylabel('avg completion time', color='brown')
axs.tick_params(axis='y', labelcolor='brown')


ax2 = axs.twinx()
mean = np.nanmean(completion_rates, axis=0)
ax2.plot(rewards,
         mean,
         linewidth=3,
         color='tab:blue')
ax2.set_ylabel('avg completion rate',
               color='tab:blue',
               rotation=270,
               labelpad=15)
ax2.tick_params(axis='y', labelcolor='tab:blue')

plt.savefig('writing/figures_thesis/vectors/planned_defections_rewards.svg',
            format='svg', dpi=300)
