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

    print(network.ctx.global_scope.variable_declarations)
    for var in network.ctx.global_scope.variable_declarations:
        print(var.typ)
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
    # for pair in initial_state.global_env:
    #     print(pair)
    # lows = None   # array of the lowest values of model variables
    # highs = None  # array of the highest values of model variables
    # total_actions = None  # total number of actions in the model
    #
    # tree = decision_tree.DecisionTree(initial_state, lows, highs, total_actions)
    #
    # current_state = initial_state
    #
    #
    # for i in range(num_of_episodes):
    #     #Algorithm 1
    #     take_action(current_state, epsilon, tree)
    #
    #     #update_tree()
    #
    #     #split_tree()
    #
    # return None

def take_action(current_state, epsilon, tree):
    action = None
    if np.random.random() < epsilon:
        action = random.choice(current_state.transitions)
    else:
        action = tree.select_action()
        pass
    #reward =
    next_state = action.destinations.pick().state
    #return action, reward, next_state

#CQI("../Testing/models/resource-gathering.v2.jani")
#CQI("../Testing/models/resource-gathering.v2_2.jani")
# CQI("../Testing/models/pacman.v1.jani")
#CQI("../Testing/models/die.jani")
#CQI("../Testing/models/firewire.true.jani")

# CQI("../Testing/models/resource-negations-not-changed.jani")
# CQI("../testing/models/resource-negations-in-expression.jani")
CQI("../testing/models/resource-working-model.jani")


