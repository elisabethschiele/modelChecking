import random

import numpy as np
from momba import engine, jani
import pathlib
import decision_tree
from src.Implementation.rewards import get_immediate_reward#, episode_finished


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
        epsilon=0.5,  # determines amount of randomness in Algorithm 2
        H_s=0.99,  # starting threshold for what potential delta_Q is required to trigger a split
        D=0.99,  # decay for H_s
        gamma=0.99,  # const for the Bellman equation
        alpha=0.01,  # const for the Bellman equation
        d=0.999,  # visit decay for Algorithm 4 and Algorithm 5
        num_of_episodes=10):
    initial_state = get_initial_state(model_path)
    # print(initial_state.global_env)
    # print(type(initial_state))

    # for boolean: Low = 0 = False
    # High = 1 = True
    lows = [1, 1, 0, 0, 0, 0, 0]  # array of the lowest values of model variables
    highs = [5, 5, 1, 1, 1, 1, 1]  # array of the highest values of model variables

    # total_actions = 4  # total number of actions in the model
    action_names = ["left", "right", "top", "down"]  # all actions
    # TODO: (optional) function that returns all actions present in the model
    # for action in initial_state.transitions:
    #     action_names.append(action.action.action_type.label)

    tree = decision_tree.DecisionTree(initial_state, lows, highs, action_names)
    new_state = initial_state

    # TODO: switch to episode_done = episode_done(new_state)


    h_s = H_s
    j = 0
    for i in range(num_of_episodes):
        new_state = initial_state
        print("****************")
        print("Episode "+str(i+1))
        print("****************")
        episode_done = False
        j = 0
        while not episode_done:
            print("Ep. "+str(i+1)+", Iter. "+str(j+1))
            # print("Struct"+tree.structure())
            # st ← current state at timestep t;
            current_state = new_state

            # L ← leaf of Tree corresponding to st;
            L = tree.root.get_leaf(current_state)
            # print("take action\n")
            # at,rt,st+1 ←TakeAction(L);
            action, reward, new_state = take_action(current_state, epsilon, tree)

            # print("update tree\n")
            # UpdateLeafQValues(L, at , rt , st+1 );
            # UpdateVisitFrequency(T ree, L);
            # UpdatePossibleSplits(L, st , at , st+1 );
            tree.update(action, reward, current_state, new_state, episode_done, alpha, gamma, d)

            # print("choose best split")

            # bestSplit, bestV value ← BestSplit(T ree, L, at)
            best_split, best_value = tree.best_split(current_state, action)
            print(f"best split: {best_split}")
            print(f"best value: {best_value}")

            # decide if we split
            # if best_value > h_s:
                # split node
            tree.split_node(current_state, L, best_split)
            # else:
            #     h_s = h_s * D

            j = j + 1
            # if j == 4:
            #     episode_done = True
            #     j = 0

            episode_done = episode_finished(new_state)
            print(f"episode_done {episode_done}")


def take_action(current_state, epsilon, tree):
    print("state: " + str(current_state.global_env))
    action = None
    if np.random.random() < epsilon:
        print("selected action randomly")
        action = random.choice(current_state.transitions)
        action_label = action.action.action_type.label
    else:
        print("selected action with biggest q value")
        # select based on largest Q-Value
        action_label = tree.select_action(current_state)

        # TODO: might be redundant. Do we only need action label?
        action = find_action_by_label(current_state, action_label)
    # print(current_state.global_env)
    print("selected action: " + action.action.action_type.label)
    new_state = action.destinations.pick().state
    reward = get_immediate_reward(current_state, new_state)
    print(f"reward {reward}")
    print("state: " + str(new_state.global_env))
    # return action, reward, new_state
    return action_label, reward, new_state

def get_value(state, variable_name):
    # returns integer value of variable_name
    switch = {
        "x": int(str(state.global_env['x'])[6:len(str(state.global_env['x'])) - 1]),
        "y": int(str(state.global_env['y'])[6:len(str(state.global_env['y'])) - 1]),
        "required_gold": int(str(state.global_env['required_gold'])[6:len(str(state.global_env['required_gold'])) - 1]),
        "required_gem": int(str(state.global_env['required_gem'])[6:len(str(state.global_env['required_gem'])) - 1]),
        "gold": int(bool(str(state.global_env['gold'])[6:len(str(state.global_env['gold'])) - 1]) == True),
        "gem": int(bool(str(state.global_env['gem'])[6:len(str(state.global_env['gem'])) - 1]) == True),
        "attacked": int(bool(str(state.global_env['attacked'])[6:len(str(state.global_env['attacked'])) - 1]) == True)
    }
    return switch.get(variable_name, None)


def episode_finished(state):
    # episode is done as soon as no more gold or gems are required
    return get_value(state, "required_gold") == 0 and get_value(state, "required_gold") == 0


def find_action_by_label(state, label):
    for action in state.transitions:
        if action.action.action_type.label == label:
            return action
    print("No action found matching label " + label)
    return -1


CQI("../Testing/models/resource-working-model.jani")
# CQI("../testing/models/resource-gathering_parsed.jani")
