import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['lines.linewidth'] = 2
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import mdp_algms

#%%
# initialising mdp for assignment submission: There is an initial state (when assignment is not started),
# potential intermediate states, and final state of completion. At each non-completed state, there is a choice
# between actions to WORK which has an immediate effort cost and SHIRK which has an immediate reward. 
# The final state is absorbing and also has a reward (equivalent to rewards from shirk). The outcome from evaluation 
# only comes at the deadline (which can be negative to positive based on state at final timepoint).  

# states of markov chain
N_intermediate_states = 1
states = np.arange(2 + N_intermediate_states) # intermediate + initial and finished states (2)

# actions available in each state 
actions = np.full(len(states), np.nan, dtype = object)
actions[:-1] = [ ['work', 'shirk'] for i in range( len(states)-1 )] # actions for all states but final
actions[-1] =  ['completed'] # actions for final state

horizon = 10 # deadline
discount_factor = 0.9 # discounting factor
efficacy = 0.9 # self-efficacy (probability of progress on working)

# utilities :
reward_pass = 4.0 
reward_fail = -4.0
reward_shirk = 0.5
effort_work = -0.4
effort_shirk = -0 
reward_completed = reward_shirk

# construct reward functions
def get_reward_functions(states, reward_pass, reward_fail, reward_shirk, reward_completed, effort_work):
    
    # reward from actions within horizon
    reward_func = np.full(len(states), np.nan, dtype = object)
    reward_func[:-1] = [ [effort_work, reward_shirk + effort_shirk] for i in range( len(states)-1 )]
    reward_func[-1] = [reward_completed]
    
    # reward from final evaluation
    reward_func_last =  np.linspace(reward_fail, reward_pass, len(states)) 
    
    return reward_func, reward_func_last

# construct transition matrix
def get_transition_prob(states, efficacy):
    
    T = np.full(len(states), np.nan, dtype = object)
    
    # for 3 states:
    T[0] = [ np.array([1-efficacy, efficacy, 0]), 
             np.array([1, 0, 0]) ] # transitions for work, shirk
    T[1] = [ np.array([0, 1-efficacy, efficacy]), 
             np.array([0, 1, 0]) ] # transitions for work, shirk
    T[2] = [ np.array([0, 0, 1]) ] # transitions for completed
    
#    # for 2 states:
#    T[0] = [ np.array([1-efficacy, efficacy]), 
#             np.array([1, 0]) ] # transitions for work, shirk
#    T[1] = [ np.array([0, 1]) ] # transitions for completed
    
    return T

#%% 
# example policy

reward_func, reward_func_last = get_reward_functions(states, reward_pass, reward_fail, reward_shirk, 
                                                     reward_completed, effort_work)
T = get_transition_prob(states, efficacy)
V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(states, actions, horizon, discount_factor, 
                              reward_func, reward_func_last, T)

# plots of policies and values
plt.figure( figsize = (8, 6) )
for i_state, state in enumerate(states):
    
    #plt.figure( figsize = (8, 6) )
    
    plt.plot(V_opt[i_state], label = 'V*%d'%i_state, marker = i_state+4, linestyle = '--')
    #plt.plot(policy_opt[i_state], label = 'policy*')
    
    for i_action, action in enumerate(actions[i_state]):
        
        plt.plot(Q_values[i_state][i_action, :], label = r'Q'+action, marker = i_state+4, linestyle = '--')
    
    #plt.title('state = %d'%state)    
    plt.legend()

#%%
# solving for policies for a range of efficacies

efficacys = np.linspace(0, 1, 50)
# optimal starting point for 4 reward regimes (change manually)
start_works = np.full( (len(efficacys), N_intermediate_states+1, 4), np.nan )

horizon = 10 # deadline
discount_factor = 0.9 # hyperbolic discounting factor
# utilities :
reward_pass = 4.0
reward_fail = -4.0
reward_shirk = 0.2
effort_work = -0.4
reward_completed = reward_shirk

for i_efficacy, efficacy in enumerate(efficacys):

    reward_func, reward_func_last = get_reward_functions(states, reward_pass, reward_fail, reward_shirk, 
                                                         reward_completed, effort_work)
    T = get_transition_prob(states, efficacy)
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(states, actions, horizon, discount_factor, 
                                  reward_func, reward_func_last, T)
    
    for i_state in range(N_intermediate_states+1):
        # find timepoints where it is optimal to work (when task not completed, state=0
        start_work = np.where( policy_opt[i_state, :] == 0 )[0]
        
        if len(start_work) > 0 :
            start_works[i_efficacy, i_state, 3] = start_work[0] # first time to start working 
            #print( policy_opt[0, :])

for i_state in range(N_intermediate_states+1):
    plt.figure(figsize=(8,6))
    legend = ['0.5:0.5', '1:0.5', '2:0.5', '4:0.5']
    for i_reward_regime, regime in enumerate(legend):
         plt.plot(efficacys, start_works[:, i_state, i_reward_regime], label = legend[i_reward_regime])
    plt.xlabel('efficacy')
    plt.ylabel('time to start work')
    plt.legend()
    plt.title('effort = %1.1f, state = %d'%(effort_work, i_state))

#%%
# solving for policies for a range of efforts

efforts = np.linspace(-8, 1, 50)
# optimal starting point for 4 reward regimes (change manually)
start_works = np.full( (len(efforts), N_intermediate_states+1, 4), np.nan ) 


horizon = 10 # deadline
discount_factor = 0.9 # hyperbolic discounting factor
# utilities :
efficacy = 0.8
reward_pass = 0.5 
reward_fail = -0.5
reward_shirk = 0.5
reward_completed = reward_shirk

for i_effort, effort_work in enumerate(efforts):

    reward_func, reward_func_last = get_reward_functions(states, reward_pass, reward_fail, reward_shirk,
                                                         reward_completed, effort_work)
    T = get_transition_prob(states, efficacy)
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(states, actions, horizon, discount_factor, 
                                  reward_func, reward_func_last, T)
    
    for i_state in range(N_intermediate_states+1):
        
        # find timepoints where it is optimal to work (when task not completed, state=0)
        start_work = np.where( policy_opt[i_state, :] == 0 )[0]
        
        if len(start_work) > 0 :
            start_works[i_effort, i_state, 0] = start_work[0] # first time to start working
            #print( policy_opt[0, :])
            
for i_state in range(N_intermediate_states+1):
    plt.figure(figsize=(8,6))
    legend = ['0.5:0.5', '1:0.5', '2:0.5', '4:0.5']
    for i_reward_regime, regime in enumerate(legend):
         plt.plot(efforts, start_works[:, i_state, i_reward_regime], label = regime)
    plt.xlabel('effort to work')
    plt.ylabel('time to start work')
    plt.legend()
    plt.title('efficacy = %1.1f state = %d' %(efficacy, i_state) )
    plt.show()
    
    
#%%
# forward runs
    
efficacys = np.linspace(0, 1, 10) # vary efficacy 
policy_always_work = np.full(np.shape(policy_opt), 0) # always work policy
V_always_work = np.full(np.shape(V_opt), 0.0)

# arrays to store no. of runs where task was finished 
count_opt = np.full( (len(efficacys), 1), 0) 
count_always_work = np.full( (len(efficacys), 1), 0)
N_runs = 5000 # no. of runs for each parameter set

for i_efficacy, efficacy in enumerate(efficacys):
    
    # get optimal policy for current parameter set
    reward_func, reward_func_last = get_reward_functions(states, reward_pass, reward_fail, reward_shirk, 
                                                         reward_completed, effort_work)
    T = get_transition_prob(states, efficacy)
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(states, actions, horizon, discount_factor, 
                                  reward_func, reward_func_last, T)
    
    # run forward (N_runs no. of times), count number of times task is finished for each policy
    initial_state = 0
    for i in range(N_runs):
         
        s, a, v = mdp_algms.forward_runs(policy_opt, V_opt, initial_state, horizon, states, T)
        if s[-1] == len(states)-1: count_opt[i_efficacy,0] +=1
        
        s, a, v = mdp_algms.forward_runs(policy_always_work, V_opt, initial_state, horizon, states, T)
        if s[-1] == len(states)-1: count_always_work[i_efficacy,0] +=1

plt.figure(figsize=(8,6))
plt.bar( efficacys, count_always_work[:, 0]/N_runs, alpha = 0.5, width=0.1, color='tab:blue', label = 'always work')
plt.bar( efficacys, count_opt[:, 0]/N_runs, alpha = 1, width=0.1, color='tab:blue', label = 'optimal policy')
plt.legend()
plt.xlabel('efficacy')
plt.ylabel('Proportion of finished runs')