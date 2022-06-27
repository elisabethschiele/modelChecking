import random

import numpy as np
from momba import engine, jani
import gym
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
    initial_states = explorer.initial_states
    (initial_state,) = initial_states
    return initial_state

def CQI(model_path,
        epsilon,              # determines amount of randomness in Algorithm 2
        H_s = 0.99,           # starting threshold for what potential delta_Q is required to trigger a split
        D = 0.99,             # decay for H_s
        gamma = 0.99,         # const for the Bellman equation
        alpha = 0.01,         # const for the Bellman equation
        d = 0.999,            # visit decay for Algorithm 4 and Algorithm 5
        num_of_episodes=10):

    initial_state = get_initial_state(model_path)
    tree = decision_tree.DecisionTree()

    current_state = initial_state



    for i in range(num_of_episodes):
        #Algorithm 1
        take_action(current_state, epsilon, q_s)

        #update_tree()

        #split_tree()

    return None

def take_action(tree, current_state, epsilon, q_s):
    action = None
    if np.random.random() < epsilon:
        action = random.choice(current_state.transitions)
    else:
        action = tree.select_action()
        pass
    #reward =
    next_state = action.destinations.pick().state
    #return action, reward, next_state
