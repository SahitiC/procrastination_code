import matplotlib.pyplot as plt
from pomdp_algms import *

#%%
# defining POMDP 

# 0: stay/check, 1: work, 2: submit
ACTIONS = ('0', '1', '2')
# 0: initial, 1: final
STATES = ('0', '1')

GAMMA = 0.95

EFFICACY = 0.8

# transition probabilities between states for each action 
T_PROB = [[[1.0, 0.0], 
           [0.0, 1.0]], 
          
          [[1.0-EFFICACY, EFFICACY], 
           [0.0, 1.0]], 
          
          [[1.0, 0.0], 
           [0.0, 1.0]]]

# OBSERVATION PROBABILITIES FOR EACH ACTION
E_PROB = [[[0.9, 0.1], 
           [0.1, 0.9]], 
          
          [[0.5, 0.5], 
           [0.5, 0.5]], 
          
          [[0.5, 0.5], 
           [0.5, 0.5]]]

# REWARDS FOR EACH ACTION IN EACH STATE
REWARDS = [[-0.1, -0.1], 
           [-3.0, -3.0], 
           [-4.0, 4.0]]

#%%

# three states

# defining POMDP 


# 0: stay/check, 1: work, 2: submit
ACTIONS = ('0', '1', '2')
# 0: initial, 1: final
STATES = ('0', '1', '2')

GAMMA = 0.95

EFFICACY = 0.8

# transition probabilities between states for each action 
T_PROB = [[[1.0, 0.0, 0.0], 
           [0.0, 1.0, 0.0],
           [0.0, 0.0, 1.0]], 
          
          [[1.0-EFFICACY, EFFICACY, 0.0], 
           [0.0, 1.0-EFFICACY, EFFICACY],
           [0.0, 0.0, 1.0]], 
          
          [[1.0, 0.0, 0.0], 
           [0.0, 1.0, 0.0],
           [0.0, 0.0, 1.0]]]

# OBSERVATION PROBABILITIES FOR EACH ACTION
E_PROB = [[[0.9, 0.05, 0.05], 
           [0.05, 0.9, 0.05],
           [0.05, 0.05, 0.9]], 
          
          [[1/3, 1/3, 1/3], 
           [1/3, 1/3, 1/3],
           [1/3, 1/3, 1/3]], 
          
          [[1/3, 1/3, 1/3], 
           [1/3, 1/3, 1/3],
           [1/3, 1/3, 1/3]]]

# REWARDS FOR EACH ACTION IN EACH STATE
REWARDS = [[-0.1, -0.1, -0.1], 
           [-3.0, -3.0, -3.0], 
           [-4.0, 0.0, 4.0]]


#%%

pomdp = POMDP(ACTIONS, T_PROB, E_PROB, REWARDS, STATES, GAMMA)

utility = pomdp_value_iteration(pomdp, epsilon=3)

#%%
colors = ['g', 'b', 'k']
for action in utility:
    for value in utility[action]:
        plt.plot(value, color=colors[int(action)])
        