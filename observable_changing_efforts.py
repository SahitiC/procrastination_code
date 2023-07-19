import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['lines.linewidth'] = 2
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import mdp_algms

#%%

# construct reward functions
def get_reward_functions(states, reward_pass, reward_fail, reward_shirk, reward_completed, 
                         effort_work, effort_shirk):
    
    # reward from actions within horizon
    reward_func = [] 
    
    for i in range( len(states)-1 ):
        reward_func.append( [effort_work[i], reward_shirk + effort_shirk]  )  # rewards in non-completed states
        
    reward_func.append( [reward_completed] ) # reward in completed state
    
    # reward from final evaluation
    reward_func_last = []
    
    for i in range( len(states)-1 ):
        reward_func_last.append( reward_fail )   # rewards in non-completed states

    reward_func_last.append(reward_pass) 
    
    return np.array(reward_func, dtype=object), np.array(reward_func_last, dtype=object)

def get_transition_prob(states, efficacy, dynamics):
    
    T = []
    T.append( [ np.array([(1-efficacy)*dynamics[0,0], (1-efficacy)*dynamics[0,1], (1-efficacy)*dynamics[0,2], efficacy]), 
                np.array([dynamics[0,0], dynamics[0,1], dynamics[0,2], 0]) ] ) # transitions for work, shirk
    
    T.append( [ np.array([(1-efficacy)*dynamics[1,0], (1-efficacy)*dynamics[1,1], (1-efficacy)*dynamics[1,2], efficacy]), 
                np.array([dynamics[1,0], dynamics[1,1], dynamics[1,2], 0]) ] ) # transitions for work, shirk
    
    T.append( [ np.array([(1-efficacy)*dynamics[2,0], (1-efficacy)*dynamics[2,1], (1-efficacy)*dynamics[2,2], efficacy]), 
                np.array([dynamics[2,0], dynamics[2,1], dynamics[2,2], 0]) ] ) # transitions for work, shirk
    
    T.append( [ np.array([0, 0, 0, 1]) ] ) # transitions for completed
    
    return np.array(T, dtype=object)

#%%

# instantiating MDP

# states of markov chain
N_DIFFICULTY_STATES = 3 # number of states of difficulty
STATES = np.arange(1 + N_DIFFICULTY_STATES) # final state and other initial states with varying difficulties of task completion

ACTIONS = []
ACTIONS = [ ['work', 'shirk'] for i in range( len(STATES)-1 ) ]
ACTIONS.append(['completed']) 

HORIZON = 10 # deadline
DISCOUNT_FACTOR = 1.0 # discounting factor
EFFICACY = 0.7# self-efficacy (probability of progress on working) in non-start/finished state

# utilities :
REWARD_PASS = 4.0 
REWARD_FAIL = -4.0
REWARD_SHIRK = 0.5 
EFFORT_WORK = [-0.2, -0.5, -1.0] # effort to complete task from one of the difficulty states
EFFORT_SHIRK = -0 
REWARD_COMPLETED = REWARD_SHIRK

# envt dynmics : transitions between difficulty states independent of actions
DYNAMICS = np.array( [[0.6, 0.2, 0.2],
                      [0.6, 0.2, 0.2],
                      [0.2, 0.6, 0.2]] ) # monotonic
#np.array( [[0.1, 0.8, 0.1],
#           [0.1, 0.1, 0.8],
#           [0.8, 0.1, 0.1]] ) # cyclic
    
#%%
# get optimal policy

reward_func, reward_func_last = get_reward_functions(STATES, REWARD_PASS, REWARD_FAIL, REWARD_SHIRK, 
                                                     REWARD_COMPLETED, EFFORT_WORK, EFFORT_SHIRK)
T = get_transition_prob(STATES, EFFICACY, DYNAMICS)
V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, 
                              reward_func, reward_func_last, T)

# plots of policies and values
colors = plt.cm.Blues(np.linspace(0.4,0.9,len(STATES)))
lines = ['--', ':']
fig, axs = plt.subplots( figsize = (8, 6) )
fig1, axs1 = plt.subplots( figsize = (5, 3) )

for i_state, state in enumerate(STATES[:-1]):
    
    
    axs.plot(V_opt[i_state], color = colors[i_state], marker = 'o', linestyle = 'None', label = f'$V^*({i_state})$',)
    axs1.plot(policy_opt[i_state], color = colors[i_state], label = f'State {i_state}')
    
    for i_action, action in enumerate(ACTIONS[i_state]):
        
        axs.plot(Q_values[i_state][i_action, :], color = colors[i_state], linestyle = lines[i_action])
        
handles, labels = axs.get_legend_handles_labels()   
handles.append(axs.plot([], [], color = 'black', linestyle = '--', label = '$Q(a=$ work$)$'))
handles.append(axs.plot([], [], color = 'black', linestyle = ':', label = '$Q(a=$ shirk$)$'))
axs.legend()
axs.set_xlabel('timesteps')
axs1.legend()
axs1.set_xlabel('timesteps')
axs1.set_yticks([0,1])
axs1.set_yticklabels(['WORK', 'SHIRK'])
axs1.set_ylabel('policy')

#%%
# forward runs

    
# run forward (N_runs no. of times), count number of times task is finished for each policy
N_runs  = 5000
initial_state = 2
for i in range(N_runs):
     
    s, a, v = mdp_algms.forward_runs(policy_opt, V_opt, initial_state, HORIZON, STATES, T)
    
