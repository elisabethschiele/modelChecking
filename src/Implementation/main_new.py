import random

import numpy as np
from graphviz import Digraph
from momba import engine, jani
import pathlib
import graphviz
from scipy import stats
import decision_tree
import decision_tree_old
import json

# TODO: change rewards file name 
from resources_rewards import get_immediate_reward

"""
TODO: 
- change model_path to string of path to desired model
- check model for variable types (only boolean, real (will be ignored) and bounded integers allowed)
- remove variables you do not want the model to consider for training (marked TODO in code)
- change get_value to work for your variables - check for correct commenting out for lakje and resources (marked TODO)
"""
model_path = "/Users/elisabeth/Desktop/model checking/modelChecking/src/Testing/models/resources_parsed_fully.jani"

def get_actions(file_path):
    labels = []
    with open(file_path, encoding="utf-8") as json_data:
        data = json.load(json_data)
        for key in data["actions"]:
            labels.append(key["name"])
    return labels

def get_lows_highs(file_path):
    # currently works with exclusively bool, real and int variables

    lows = []
    highs = []
    var_labels = []
    with open(file_path, encoding="utf-8") as json_data:
        data = json.load(json_data)
        for key in data["variables"]:
            print(key)
            if key["type"] == "bool":
                print("detected bool")
                lows.append(0)
                highs.append(1)
                var_labels.append(key["name"])
            elif key["type"] == "real":
                pass
            else:
                lows.append(key["type"]["lower-bound"])
                highs.append(key["type"]["upper-bound"])
                var_labels.append(key["name"])
    return lows, highs, var_labels

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
        epsilon=0.5,   # determines amount of randomness in Algorithm 2
        H_s=1,  # starting threshold for what potential delta_Q is required to trigger a split
        D=0.999,  # decay for H_s
        gamma=0.99,  # const for the Bellman equation
        alpha=0.1,  # const for the Bellman equation 0.01
        d=0.999,  # visit decay for Algorithm 4 and Algorithm 5
        num_of_episodes=5000, 
        num_of_steps = 500000):
    initial_state = get_initial_state(model_path)
    
    # automatically extract variable bounds and labels
    lows, highs, var_labels = get_lows_highs(model_path)

    # TODO: remove variables you don't want the model to consider
    # For resources model we do not want the variable "success" to have an impact on our training
    lows.pop(0)
    highs.pop(0)
    var_labels.pop(0)

    action_names = get_actions(model_path)  # all actions
    
    # initialize tree
    tree = decision_tree.DecisionTree(initial_state, lows, highs, action_names, var_labels)
    new_state = initial_state

    iters_per_episode = []
    avg_reward_per_episode = []
    leaf_count = []
    h_s = H_s
    step = 0
    for i in range(num_of_episodes):
        new_state = initial_state
        print("Episode "+str(i+1))
        episode_done = False
        j = 0
        h_s=H_s

        total_episode_reward = 0

        while not episode_done:
            step += 1

            # st ← current state at timestep t;
            current_state = new_state

            # L ← leaf of Tree corresponding to st;
            L = tree.root.get_leaf(current_state)
            # at,rt,st+1 ←TakeAction(L);
            action, reward, new_state = take_action(current_state, epsilon, tree, step, num_of_steps)
            total_episode_reward += reward
            # UpdateLeafQValues(L, at , rt , st+1 );
            # UpdateVisitFrequency(T ree, L);
            # UpdatePossibleSplits(L, st , at , st+1 );
            tree.update(action, reward, current_state, new_state, episode_done, alpha, gamma, d)
            # bestSplit, bestV value ← BestSplit(T ree, L, at)
            best_split, best_value = tree.best_split(current_state, action)

            # decide if we split
            if best_value > h_s:
                # split node
                tree.split_node(current_state, L, best_split)
                h_s = H_s
            else:
                h_s = h_s * D

            j = j + 1

            episode_done = episode_finished(new_state)
            if episode_done:
                iters_per_episode.append(j)
                avg_reward_per_episode.append(total_episode_reward / j)
                leaf_count.append(tree.get_total_leaves())

            if tree.get_total_leaves() == 1000:
                break

            if step == num_of_steps:
                break
        if step == num_of_steps:
            break
        if tree.get_total_leaves() == 1000:
            break
    g = Digraph('G', filename='graph.gv')
    tree.plot(g)
    print(g.source)
    print(f'iters per episode: {str(iters_per_episode)}')
    print(f'avg reward per episode: {avg_reward_per_episode}')
    print(f'total leaves per episode: {leaf_count}')
    print(f'tot. leaves/pos. leaves: {tree.get_total_leaves()}/{total_possible_leaves(lows, highs)}')

    # STATS
    save_stat_avg_reward("../Testing/Simulations/avg_rewards.txt", avg_reward_per_episode)




    # eps_func = (lambda step: max(0.1, 1 - step / 2 * 1e2))
    # print(f'eps_func(1) = {eps_func(1)}')
    # print(f'eps_func(2) = {eps_func(2)}')
    # print(f'eps_func(3) = {eps_func(3)}')
    # print(f'eps_func(4) = {eps_func(4)}')
    # print(f'eps_func(5) = {eps_func(5)}')
    # print(f'eps_func(6) = {eps_func(6)}')
    # print(f'eps_func(7) = {eps_func(7)}')
    # print(f'eps_func(8) = {eps_func(8)}')
    # print(f'eps_func(10) = {eps_func(10)}')
    # print(f'eps_func(100) = {eps_func(100)}')
    # print(f'eps_func(1000) = {eps_func(1000)}')
    # print(f'eps_func(10000) = {eps_func(10000)}')
    # print(f'eps_func(50000) = {eps_func(50000)}')

def take_action(current_state, epsilon, tree, step, num_of_steps):
    action = None

    eps_func = (lambda step: max(0.05, 1 - step / (num_of_steps)))
    if np.random.random() < eps_func(step):
        # select action randomly
        action = random.choice(current_state.transitions)
        action_label = action.action.action_type.label
    else:
        # select based on largest Q-Value
        action_label = tree.select_action(current_state)

        action = find_action_by_label(current_state, action_label)

    new_state = action.destinations.pick().state
    reward = get_immediate_reward(current_state, new_state)

    return action_label, reward, new_state

def find_action_by_label(state, label):
    for action in state.transitions:
        if action.action.action_type.label == label:
            return action
    print("No action found matching label " + label)
    return -1

def total_possible_leaves(lows, highs):
    total = 1
    for i in range(len(lows)):
        total *= highs[i] - lows[i] + 1
    return total

def save_stat_avg_reward(filename, avg_reward):
    f = open(filename, "w")
    for i in range(len(avg_reward)):
        f.write(str(i+1)+" "+str(avg_reward[i])+"\n")


# TODO: change to correct get_value function

def get_value(state, variable_name):
    # returns integer value of variable_name
    switch = {
        "x": int(str(state.global_env['x'])[6:len(str(state.global_env['x'])) - 1]),
        "y": int(str(state.global_env['y'])[6:len(str(state.global_env['y'])) - 1]),
        "required_gold": int(str(state.global_env['required_gold'])[6:len(str(state.global_env['required_gold'])) - 1]),
        "required_gem": int(str(state.global_env['required_gem'])[6:len(str(state.global_env['required_gem'])) - 1]),
        "gold": int(str(state.global_env['gold'])[6:len(str(state.global_env['gold'])) - 1] == "True"),
        "gem": int(str(state.global_env['gem'])[6:len(str(state.global_env['gem'])) - 1] == "True"),
        "attacked": int(str(state.global_env['attacked'])[6:len(str(state.global_env['attacked'])) - 1] == "True")
    }
    return switch.get(variable_name, None)

# def get_value(state, variable_name):
#     # DONE

#     # returns integer value of variable_name
#     switch = {
#         "r": int(str(state.global_env['r'])[6:len(str(state.global_env['r'])) - 1]),
#         "c": int(str(state.global_env['c'])[6:len(str(state.global_env['c'])) - 1]),
#            }
#     return switch.get(variable_name, None)


# TODO: change to correct finishing condition
def episode_finished(state):
    # episode is done as soon as no more gold or gems are required
    return get_value(state, "required_gold") == 0 and get_value(state, "required_gem") == 0

# def episode_finished(state):
#     # episode is done if goal is reached
#     return (get_value(state, "r") == 0 and get_value(state, "c") == 5)


# ------------------------------------------------ End of new algorithm ------------------------------------------------

def Old_Alg(model_path,
        epsilon=0.5,  # determines amount of randomness in Algorithm 2
        H_s=1,  # starting threshold for what potential delta_Q is required to trigger a split
        D=0.999,  # decay for H_s
        gamma=0.99,  # const for the Bellman equation
        alpha=0.1,  # const for the Bellman equation 0.01
        d=0.999,  # visit decay for Algorithm 4 and Algorithm 5
        hist_min_size = 200,
        num_of_episodes=300,
        num_of_steps = 10000):
    initial_state = get_initial_state(model_path)

    lows = [1, 1, 0, 0, 0, 0, 0]  # array of the lowest values of model variables
    highs = [5, 5, 1, 1, 1, 1, 1]  # array of the highest values of model variables

    action_names = ["left", "right", "top", "down"]  # all actions
    tree = decision_tree_old.DecisionTreeOld(initial_state, lows, highs, action_names)
    new_state = initial_state

    iters_per_episode = []
    h_s = H_s
    step = 0
    split_count = 0
    for i in range(num_of_episodes):
        new_state = initial_state
        print("****************")
        print("Episode "+str(i+1))
        print("****************")
        print("state: " + str(new_state.global_env))
        episode_done = False
        j = 0
        #h_s=H_s
        while not episode_done:
            step += 1
            print("Ep. "+str(i+1)+", Iter. "+str(j+1))
            j += 1
            print("Struct"+tree.structure())
            # st ← current state at timestep t;
            current_state = new_state

            # L ← leaf of Tree corresponding to st;
            L = tree.root.get_leaf(current_state)
            # at,rt,st+1 ←TakeAction(L);
            action, reward, new_state, random_action = take_action_old(current_state, epsilon, tree, step, num_of_steps)

            # split = True if a split needs to be performed
            split = tree.update(action, reward, current_state, new_state, episode_done, alpha, gamma, d, random_action, hist_min_size)
            print(f'split: {split}')

            if split:

                tree.split_node(current_state, L)
                split_count += 1
                if split_count == 2:
                    print("SECOND SPILT")
                    break

            episode_done = episode_finished(new_state)
            if episode_done:
                iters_per_episode.append(j)

            if step == num_of_steps:#TODO: experiment
                break
        if step == num_of_steps:#TODO: experiment
            break
        if split_count == 2:
            break
    g = Digraph('G', filename='graph.gv')
    tree.plot(g)
    print(g.source)
    print(f'iters per episode: {str(iters_per_episode)}')

    a = [10, 11, 12, 13, 14, 15]
    b = [5, 5]
    # a1 = [10, 11, 12, 13, 14, 15]
    # b1 = [-10, -11, -14]
    # print("T-test: "+str(stats.ttest_ind(a, b)))
    # print("T-test: " + str(stats.ttest_ind(a1, b1)))

def take_action_old(current_state, epsilon, tree, step, num_of_steps):

    # eps_func = (lambda step: max(0.05, 1 - step / (num_of_steps)))
    # print(f'prob_random {step} = {eps_func(step)}')
    if np.random.random() < epsilon:
    # if np.random.random() < eps_func(step):
        print("selected action randomly")
        random_action = True # True if action was chosen randomly
        action = random.choice(current_state.transitions)
        action_label = action.action.action_type.label
    else:
        print("selected action with biggest q value")
        random_action = False
        # select based on largest Q-Value
        action_label = tree.select_action(current_state)
        action = find_action_by_label(current_state, action_label)
    # print(current_state.global_env)
    print("selected action: " + action.action.action_type.label)

    new_state = action.destinations.pick().state
    reward = get_immediate_reward(current_state, new_state)
    print(f"reward {reward}")
    print("new_state: " + str(new_state.global_env))
    print(f"value of gold: {get_value(new_state, 'gold')}")
    # return action, reward, new_state, random_action
    return action_label, reward, new_state, random_action

CQI(model_path)
# Old_Alg("../Testing/models/resource-working-model.jani")
# CQI("../testing/models/resource-gathering_parsed.jani")

