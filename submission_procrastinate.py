"""
Defining POMDP problem where assignment submission is procrastinated. Once a 
completion state is reached, there is a chance that a new idea might strike to make the 
assignment better. Whether it is possible can be found only by checking (thinking, collecting info).
This file format is for the pomdp_py package.
"""
import pomdp_py

class State(pomdp_py.State):
    
    def __init__(self, name):
        if name != "initial" and name != "finish_0"\
            and name != "finish_1" and name != "final":
            raise ValueError("Invalid state: %s" % name)
        self.name = name

class Action(pomdp_py.Action):
    def __init__(self, name):
        if name != "check" and name != "work"\
           and name != "submit":
            raise ValueError("Invalid action: %s" % name)
        self.name = name

class Observation(pomdp_py.Observation):
    def __init__(self, name):
        if name != "initial" and name != "finish_0"\
            and name != "finish_1" and name != "final":
            raise ValueError("Invalid action: %s" % name)
        self.name = name
    
    
class ObservationModel(pomdp_py.ObservationModel):
    
    def __init__(self, noise=0.15):
        self.noise = noise

    def probability(self, observation, next_state, action):
        
        if action.name == "check":
            
            if next_state.name == "finish_0" or next_state.name == 'finish_1':
                if observation.name == next_state.name:
                    return 1.0 - self.noise  # got correct info
                elif observation.name == 'initial' or observation.name == 'final':
                    return 0.0
                else:
                    return self.noise
                
            elif next_state.name == "initial" or next_state.name == 'final':
                if observation.name == next_state.name:
                    return 1.0   # full info
                else:
                    return 0.0
                
        else: # for work and submit
            
            if next_state.name == "finish_0" or next_state.name == 'finish_1':
                if observation.name == "finish_0" or observation.name == "finish_1":
                    return 0.5  # got correct info
                elif observation.name == 'initial' or observation.name == 'final':
                    return 0.0
                
            elif next_state.name == "initial" or next_state.name == 'final':
                if observation.name == next_state.name:
                    return 1.0   # full info (deterministic)
                else:
                    return 0.0

    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation
        space (e.g. value iteration)"""
        return [Observation(s)
                for s in {"initial", "finish_0", "finish_1", "final"}]
    
    
class TransitionModel(pomdp_py.TransitionModel):
    
    def __init__(self, efficacy=0.9):
        self.efficacy = efficacy
        
    def probability(self, next_state, state, action):
        
        if action.name == "work":
            
            if state.name == "initial":
                if next_state.name == "initial": 1.0 - self.efficacy 
                elif next_state.name == "finish_0": self.efficacy * 0.5
                elif next_state.name == "finish_1": self.efficacy * 0.5
                else: 0.0 
            
            elif state.name == "finish_0":
                if next_state.name == state.name:
                    return 1.0 - 0.0
                else:
                    return 0.0
                
            elif state.name == "finish_1":
                if next_state.name == "initial": 0.0 
                elif next_state.name == "finish_0": 0.0
                elif next_state.name == "finish_1": 1.0 - self.efficacy 
                elif next_state.name == "final": self.efficacy
                
            elif state.name == "final":
                if next_state.name == state.name:
                    return 1.0 - 0.0
                else:
                    return 0.0
            
        # for submit and check actions (identity)        
        else:
            if next_state.name == state.name:
                return 1.0 - 0.0
            else:
                return 0.0


    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the
        observation space (e.g. value iteration)"""
        return [State(s) for s in {"initial", "finish_0", "finish_1", "final"}]
    
    def sample(self, state, action):
        return State(state.name)
        
 
class RewardModel(pomdp_py.RewardModel):
    
    def __init__(self, check_effort, work_effort, reward_fail, reward_pass, reward_bonus):
        self.check_effort = check_effort
        self.work_effort = work_effort
        self.reward_fail = reward_fail
        self.reward_pass = reward_pass
        self.reward_bonus = reward_bonus
    
    def _reward_func(self, state, action):
        
        if action.name == "check":
            return self.check_effort
        
        elif action.name == "work":
            return self.work_effort

        elif action.name == "submit":
            if state.name == "initial": self.reward_fail
            elif state.name == "finish_0": self.reward_pass
            elif state.name == "finish_1": self.reward_pass
            elif state.name == "final": self.reward_bonus
            

class SubmitProblem(pomdp_py.POMDP):

    def __init__(self, obs_noise, efficacy,
                 check_effort, work_effort, reward_fail, reward_pass, reward_bonus,
                 init_true_state):
        """init_belief is a Distribution."""
        agent = pomdp_py.Agent(TransitionModel(efficacy),
                               ObservationModel(obs_noise),
                               RewardModel(check_effort, work_effort, reward_fail, reward_pass, reward_bonus))
        env = pomdp_py.Environment(init_true_state,
                                   TransitionModel(),
                                   RewardModel())
        super().__init__(agent, env, name="SubmitProblem")