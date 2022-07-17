import random

import numpy as np
from momba import engine, jani
#import gym
import pathlib
import decision_tree

def get_initial_state(model_path):
    path = pathlib.Path(model_path)
    network = jani.load_model(path.read_text("utf-8"))
    explorer = engine.Explorer.new_discrete_time(
        network,
        parameters={
            "delay": 3,
            "deadline": 200,
        }
    )

    # print(network.ctx.global_scope.variable_declarations)
    # for var in network.ctx.global_scope.variable_declarations:
        # print(var.typ)
        # if var.typ ==

    initial_states = explorer.initial_states
    (initial_state,) = initial_states
    return initial_state


def CQI(model_path,
        epsilon = 0.5,        # determines amount of randomness in Algorithm 2
        H_s = 0.99,           # starting threshold for what potential delta_Q is required to trigger a split
        D = 0.99,             # decay for H_s
        gamma = 0.99,         # const for the Bellman equation
        alpha = 0.01,         # const for the Bellman equation
        d = 0.999,            # visit decay for Algorithm 4 and Algorithm 5
        num_of_episodes=10):

    

    initial_state = get_initial_state(model_path)
    # print(initial_state.global_env)
    # print(type(initial_state))

    # for boolean: Low = 0 = False
    # High = 1 = True
    lows = [1, 1, 0, 0, 0, 0, 0]   # array of the lowest values of model variables
    highs = [5, 5, 1, 1, 1, 1, 1]  # array of the highest values of model variables
    total_actions = 4  # total number of actions in the model
    

    tree = decision_tree.DecisionTree(initial_state, lows, highs, total_actions)
    current_state = initial_state
    next_state = initial_state

    episode_done = False

    for i in range(num_of_episodes):
        while not episode_done:
            # st ← current state at timestep t;
            current_state = next_state
            
            # L ← leaf of Tree corresponding to st;
            L = tree.root.get_leaf(current_state)
            # at,rt,st+1 ←TakeAction(L);
            a = take_action(current_state, epsilon, tree)
            r = 0 # TODO
            # next_state = a.destinations.pick().state
            # UpdateLeafQValues(L, at , rt , st+1 );
            # UpdateVisitFrequency(T ree, L);
            # UpdatePossibleSplits(L, st , at , st+1 );
            
            # update_tree()

            # split_tree()
            

            episode_done = True
    
def take_action(current_state, epsilon, tree):
    action = None
    if np.random.random() < epsilon:
        action = random.choice(current_state.transitions)
    else:
        action = tree.select_action(current_state)
        pass
    return action

def get_value(state, variable_name):
    # returns integer value of variable_name
    switch={
        "x": int(str(state.global_env['x'])[6:len(str(state.global_env['x']))-1]),
        "y": int(str(state.global_env['y'])[6:len(str(state.global_env['y']))-1]),
        "required_gold": int(str(state.global_env['required_gold'])[6:len(str(state.global_env['required_gold']))-1]),
        "required_gem": int(str(state.global_env['required_gem'])[6:len(str(state.global_env['required_gem']))-1]),
        "gold": int(bool(str(state.global_env['gold'])[6:len(str(state.global_env['gold']))-1]) == True),
        "gem": int(bool(str(state.global_env['gem'])[6:len(str(state.global_env['gem']))-1]) == True),
        "attacked": int(bool(str(state.global_env['attacked'])[6:len(str(state.global_env['attacked']))-1]) == True)
    }
    return switch.get(variable_name, None)

def episode_done(state):
    # episode is done as soon as no more gold or gems are required
    return get_value(state, "required_gold") == 0 and get_value(state, "required_gold") == 0


CQI("../testing/models/resource-working-model.jani")

