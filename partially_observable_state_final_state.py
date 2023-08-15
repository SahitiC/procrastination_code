import numpy as np
import matplotlib.pyplot as plt
import math

#%%
def pO_ba(belief, action, observation, e_prob, t_prob):
    """
    probability of observation given the belief and action
    """
    return e_prob[action][:, observation].T @ (t_prob[action].T @ belief)

def get_next_belief(belief, action, observation, e_prob, t_prob):
    """
    finding the next belief state given belief, action and observation
    """
    return e_prob[action][:, observation] * (t_prob[action].T @ belief) / (
           pO_ba(belief, action, observation, e_prob, t_prob) )

def round_down(num,digits):
    factor = 10.0 ** digits
    return math.floor(num * factor) / factor

def round_up(num,digits):
    factor = 10.0 ** digits
    return math.ceil(num * factor) / factor

def interpolate1d(x, x_0, x_1, dx,func_0,func_1):
    """
    interpolate for a discrete 1D function
    """
    interpolated_func = 0.0
    interpolated_func = (((x-x_0)*func_1) + ((x_1-x)*func_0))/dx
    return interpolated_func

def interpolate2d(x, x_0, x_1, dx, y, y_0, y_1, dy,
                  funcx0_y0,funcx1_y0,funcx0_y1,funcx1_y1):
    """
    interpolate for a discrete 2D function using 1D-interoplation 
    """
    interpolate_y0 = interpolate1d(x,x_0,x_1,dx,funcx0_y0, funcx1_y0)
    interpolate_y1 = interpolate1d(x,x_0,x_1,dx,funcx0_y1, funcx1_y1)
    interpolated_func = interpolate1d(y,y_0,y_1,dy,interpolate_y0,interpolate_y1)
    return interpolated_func

def interpolate_value(belief, value, db):
    """
    interpolate value for an arbitrary belief given a discretised value function on a belief grid
    """
    mult = 1/db
    return interpolate2d(belief[0], round_down(belief[0], 2), round_up(belief[0], 2), db,
                         belief[1], round_down(belief[1], 2), round_up(belief[1], 2), db,
                         value[int(round_down(belief[0], 2)*mult), int(round_down(belief[0], 2)*mult)],
                         value[int(round_up(belief[0], 2)*mult), int(round_down(belief[0], 2)*mult)],
                         value[int(round_down(belief[0], 2)*mult), int(round_up(belief[0], 2)*mult)],
                         value[int(round_up(belief[0], 2)*mult), int(round_up(belief[0], 2)*mult)])
    
    
#%%
# define the pomdp 
states = np.array( [0, 1, 2] ) # (1,0), (1,1), 2
actions = np.array( ['check', 'work', 'submit'])
observations = np.array( [0, 1, 2] )
# transition probabilities between states for each action 
efficacy = 0.9
noise = 0.2
discount_factor = 0.95
db = 0.01 # discretisation of belief space
# transition probabilities between states for each action 
t_prob = np.array( [[[1.0, 0.0, 0.0], 
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]], 
          
                    [[1.0, 0.0, 0.0], 
                     [0.0, 1.0-efficacy, efficacy],
                     [0.0, 0.0, 1.0]], 
          
                     [[0.5, 0.5, 0.0], 
                      [0.5, 0.5, 0.0],
                      [0.5, 0.5, 0.0]]] )

# observation probabilities for each action
e_prob =  np.array( [[[1.0-noise, noise, 0.0], 
                    [noise, 1.0-noise, 0.0],
                    [0.0, 0.0, 1.0]], 
                  
                   [[0.5, 0.5, 0.0], 
                    [0.5, 0.5, 0.0],
                    [0.0, 0.0, 1.0]], 
                  
                   [[0.5, 0.5, 0.0], 
                    [0.5, 0.5, 0.0],
                    [0.0, 0.0, 1.0]]] )

# rewards for each action in each state
rewards = np.array([[-0.1, -0.1, -0.1], 
                    [-1.0, -1.0, -1.0], 
                    [-4.0, 0.0, 4.0]])

#%%

# define value and policy variables in grid of belief states
# represent the beliefs of first N-1 states (N: total number of states)
value = np.full( (int(1/db)+1, int(1/db)+1), np.nan )
policy = np.full( (int(1/db)+1, int(1/db)+1), np.nan )
b = np.arange(0, 1+db, db)

# initialise value function 
for i_b1 in range(len(b)):
    
    for i_b2 in range(len(b)):
        
        if b[i_b1] + b[i_b2] <= 1.0:
            value[i_b1, i_b2] = 0.0
            
i_iter = 0 # number of value iterations
value_new = value
for i_b1 in range(len(b)):
    
    for i_b2 in range(len(b)):
        
        if b[i_b1] + b[i_b2] <= 1.0:
            
            belief = np.array( [b[i_b1], b[i_b2], 1-b[i_b1]-b[i_b2]] )
            
            q = np.zeros((len(actions),)) # q-value for each action
            
            for i_action, action in enumerate(actions):
                
                future_value = 0.0
                
                for i_observation, observation in enumerate(observations):
                    
                    next_belief = get_next_belief(belief, action, observation, e_prob, t_prob)
                    future_value = future_value + \
                                   pO_ba(belief, action, observation, e_prob, t_prob) * \
                                   interpolate_value(next_belief, value, db)
                    
                q[i_action] = belief.T @ rewards[i_action, :] + discount_factor * future_value
                              
            value_new[i_b1, i_b2] = np.max(q)
            policy[i_b1, i_b2] = np.argmax(q)
                
            

    

