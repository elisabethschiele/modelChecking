import random
import numpy as np
import json
from graphviz import Digraph
from momba import engine, jani
import pathlib

import decision_tree_new
import decision_tree_old

# Important! Change the following import depending on your model
from resources_rewards import get_immediate_reward, episode_finished
# from lake_rewards import get_immediate_reward, episode_finished

"""
This is the main file in which the high-level structure of the both algorithms is defined
and all the simulations are performed.
This file performs all tests on the resources_parsed_fully.jani file
To change the amount of gold and gems to collect change the model variables and their bounds within the JANI file
"""


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
            if key["type"] == "bool":
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
        epsilon=0.5,  # determines amount of randomness in Algorithm 2
        H_s=8,  # starting threshold for what potential delta_Q is required to trigger a split
        D=0.9999,  # decay for H_s
        gamma=0.8,  # const for the Bellman equation 0.99
        alpha=0.1,  # const for the Bellman equation 0.01
        d=0.999,  # visit decay for Algorithm 4 and Algorithm 5
        num_of_episodes=10000,
        num_of_steps=1000000):
    """
    CQI Algorithm from the new paper
    """

    initial_state = get_initial_state(model_path)
    lows, highs, var_labels = get_lows_highs(model_path)

    """
    Note: some models, like Resource gathering include variables that cannot be accessed via state.global_env
    with Momba. There are typically the variables that are not relevant for splitting, since they normally  
    represent some function values (like whether the model is in a state that reached a final goal). We need to 
    manually exclude them. In case of Resource gathering we exclude the variable "success" with the three following
    lines. 
    
    Important! Comment/adjust them for model used as needed.
    """

    lows.pop(0)
    highs.pop(0)
    var_labels.pop(0)

    action_names = get_actions(model_path)  # all actions
    tree = decision_tree_new.DecisionTreeNew(initial_state, lows, highs, action_names, var_labels)

    iters_per_episode = []
    avg_reward_per_episode = []
    leaf_count = []
    node_count = []
    h_s = H_s
    step = 0
    for i in range(num_of_episodes):
        new_state = initial_state
        print("Episode " + str(i + 1))
        episode_done = False
        j = 0
        h_s = H_s
        total_episode_reward = 0

        while not episode_done:
            step += 1
            # st ← current state at timestep t;
            current_state = new_state

            # L ← leaf of Tree corresponding to st;
            L = tree.root.get_leaf(current_state)

            # at,rt,st+1 ←TakeAction(L);
            action, reward, new_state = take_action(current_state, epsilon, tree, step, num_of_steps, True)
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
                node_count.append(tree.get_total_nodes())

            if tree.get_total_leaves() == 1000:
                break

            if step == num_of_steps:
                break
        if step == num_of_steps:
            break
        if tree.get_total_leaves() == 1000:
            break

    g = Digraph('G', filename='graph.gv')  # generate graph in graphviz format
    tree.plot(g)
    print(g.source)
    print(f'iters per episode: {str(iters_per_episode)}')
    print(f'avg reward per episode: {avg_reward_per_episode}')
    print(f'total leaves per episode: {leaf_count}')
    print(f'total nodes per episode: {node_count}')
    print(f'tot. leaves/pos. leaves: {tree.get_total_leaves()}/{total_possible_leaves(lows, highs)}')
    print(f'tot. nodes: {tree.get_total_nodes()}')

    it_main = iters_per_episode
    rew_main = avg_reward_per_episode
    lc_main = leaf_count
    nc_main = node_count

    # uncomment for testing - evaluation on additional k steps without changing the learned tree further
    # it, rew, lc, nc = evaluate(model_path, tree, initial_state, D, gamma, alpha, d, num_of_episodes, num_of_steps)

    # reinitialize global variables
    tree.reinit_nodes()
    tree.reinit_leaves()
    tree.reinit_ids()

    # uncomment for testing
    return it_main, rew_main, lc_main, nc_main  # , it, rew, lc, nc


def evaluate(model_path,
             tree,
             initial_state,
             H_s=8,  # starting threshold for what potential delta_Q is required to trigger a split
             num_of_episodes=10000,
             num_of_steps=50000,
             ):
    """
    The function runs :param num_of_steps steps without performing any
    updates on the tree for evaluation purposes. Used for testing.
    """

    epsilon = 0.05

    iters_per_episode = []
    avg_reward_per_episode = []
    leaf_count = []
    node_count = []
    step = 0
    for i in range(num_of_episodes):
        new_state = initial_state
        print("Episode " + str(i + 1))
        episode_done = False
        j = 0
        h_s = H_s

        total_episode_reward = 0

        while not episode_done:
            step += 1
            current_state = new_state
            action, reward, new_state = take_action(current_state, epsilon, tree, step, num_of_steps, False)
            total_episode_reward += reward
            j = j + 1

            episode_done = episode_finished(new_state)
            if episode_done:
                iters_per_episode.append(j)
                avg_reward_per_episode.append(total_episode_reward / j)
                leaf_count.append(tree.get_total_leaves())
                node_count.append(tree.get_total_nodes())

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
    print(f'total nodes per episode: {node_count}')
    # print(f'tot. leaves/pos. leaves: {tree.get_total_leaves()}/{total_possible_leaves(lows, highs)}')

    # collect relevant statistics
    avg_iters_per_episode = sum(iters_per_episode) / len(iters_per_episode)
    avg_avg_reward_per_episode = sum(avg_reward_per_episode) / len(avg_reward_per_episode)

    return avg_iters_per_episode, avg_avg_reward_per_episode, tree.get_total_leaves(), tree.get_total_nodes()


def take_action(current_state, epsilon, tree, step, num_of_steps, epsilon_decaying):
    # action = None
    eps_func = (lambda step: max(0.05, 1 - step / (num_of_steps)))
    if epsilon_decaying:
        prob_random = eps_func(step)
    else:
        prob_random = epsilon
    if np.random.random() < prob_random:
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


def hs_sim_res_gath(filename, algorithm, model):
    """
    Simulation for how different splitting threshold affect the best achieved rewards and tree size.
    """
    hs = 0.1
    f = open(filename, "w")
    while hs < 1300:
        _, rew, lc, nc = algorithm(model, H_s=hs)
        print(f'hs: {hs}, rew: {max(rew)}, lc: {max(lc)}, nc: {max(nc)}')
        f.write(str(hs) + " " + str(max(rew)) + " " + str(max(lc)) + " " + str(max(nc)) + "\n")
        hs *= 1.1
    f.close()


def save_full_stats_res_gath(filename, runs, algorithm, model):
    """
    Simulation for determining how the avg. reward per episode and the tree size change overtime.
    We perform 10 runs of the algorithm and save the average values for each episode. We throw away
    (but count and display) the runs which did not found a good policy (see :method alg_converged()).
    The results are saved in :param filename.
    """

    arr_iters_per_episode = []
    arr_avg_reward_per_episode = []
    arr_leaf_count = []
    arr_node_count = []

    its = []
    rews = []
    lcs = []
    ncs = []

    i = runs
    not_converged = 0
    while i != 0:
        iters_per_episode, avg_reward_per_episode, leaf_count, node_count, it, rew, lc, nc = algorithm(model)

        print(f'converged: {alg_converged(iters_per_episode)}')
        # check for convergence:
        if not alg_converged(iters_per_episode):
            not_converged += 1
            continue

        arr_iters_per_episode.append(iters_per_episode)
        arr_avg_reward_per_episode.append(avg_reward_per_episode)
        arr_leaf_count.append(leaf_count)
        arr_node_count.append(node_count)
        its.append(it)
        rews.append(rew)
        lcs.append(lc)
        ncs.append(nc)
        i -= 1

    min_size = min(map(len, arr_iters_per_episode))
    avg_iters_per_episode = []
    avg_avg_reward_per_episode = []
    avg_leaf_count = []
    avg_node_count = []

    for i in range(min_size):
        sum_iter = 0
        sum_reward = 0
        sum_leaves = 0
        sum_nodes = 0
        for j in range(runs):
            sum_iter += arr_iters_per_episode[j][i]
            sum_reward += arr_avg_reward_per_episode[j][i]
            sum_leaves += arr_leaf_count[j][i]
            sum_nodes += arr_node_count[j][i]
        avg_iters_per_episode.append(sum_iter / runs)
        avg_avg_reward_per_episode.append(sum_reward / runs)
        avg_leaf_count.append(sum_leaves / runs)
        avg_node_count.append(sum_nodes / runs)

    print(
        f'size_1 {len(arr_iters_per_episode[0])}, size_2 {len(arr_iters_per_episode[1])}, size_3 {len(arr_iters_per_episode[2])}')
    print(f'avg_iters: {avg_iters_per_episode}')
    print(f'avg_rewards: {avg_avg_reward_per_episode}')
    print(f'avg_leaves: {avg_leaf_count}')
    print(f'avg_nodes: {avg_node_count}')
    print(f'not converged: {not_converged}')

    # Results for evaluation: avg. values and std. deviations. Note, that these are not
    # saved in file, but rather printed to the console directly.
    print(f'avg it: {sum(its) / len(its)}, std. dev.: {np.std(its)}')
    print(f'avg rew: {sum(rews) / len(rews)}, std. dev.: {np.std(rews)}')
    print(f'avg lc: {sum(lcs) / len(lcs)}, std. dev.: {np.std(lcs)}')
    print(f'avg nc: {sum(ncs) / len(ncs)}, std. dev.: {np.std(ncs)}')
    print(f'iters: {its}')
    print(f'rews: {rews}')

    f = open(filename, "w")
    for i in range(min_size):
        f.write(str(i + 1) + " " + str(avg_iters_per_episode[i]) + " " + str(avg_avg_reward_per_episode[i]) + " " + str(
            avg_leaf_count[i]) + " " + str(avg_node_count[i]) + "\n")
    f.close()


def alg_converged(iters_per_episode):
    """
    Determines if the algorithm has most likely found a good policy.
    """
    return sum(iters_per_episode[:20]) > sum(iters_per_episode[-20:]) * 3


def Old_Alg(model_path,
            epsilon=0.5,  # determines amount of randomness in Algorithm 2
            gamma=0.8,  # const for the Bellman equation
            alpha=0.3,  # const for the Bellman equation 0.01
            d=0.999,  # visit decay for Algorithm 4 and Algorithm 5
            hist_min_size=3000,
            num_of_episodes=10000,
            num_of_steps=1000000):
    """
    Pyeatt Algorithm from the old paper.
    """

    initial_state = get_initial_state(model_path)
    lows, highs, var_labels = get_lows_highs(model_path)

    """
    Note: some models, like Resource gathering include variables that cannot be accessed via state.global_env
    with Momba. There are typically the variables that are not relevant for splitting, since they normally  
    represent some function values (like whether the model is in a state that reached a final goal). We need to 
    manually exclude them. In case of Resource gathering we exclude the variable "success" with the three following
    lines. 

    Important! Comment/adjust them for model used as needed.
    """

    lows.pop(0)
    highs.pop(0)
    var_labels.pop(0)

    # action_names = ["left", "right", "top", "down"]  # all actions
    action_names = get_actions(model_path)
    tree = decision_tree_old.DecisionTreeOld(initial_state, lows, highs, action_names, var_labels)

    iters_per_episode = []
    avg_reward_per_episode = []
    leaf_count = []
    node_count = []
    step = 0
    split_count = 0
    for i in range(num_of_episodes):
        new_state = initial_state
        print("Episode " + str(i + 1))
        episode_done = False
        j = 0
        total_episode_reward = 0
        while not episode_done:
            step += 1
            j += 1
            # st ← current state at timestep t;
            current_state = new_state

            # L ← leaf of Tree corresponding to st;
            L = tree.root.get_leaf(current_state)

            # at,rt,st+1 ←TakeAction(L);
            action, reward, new_state, random_action = take_action_old(current_state, epsilon, tree, step, num_of_steps)
            total_episode_reward += reward

            # split = True if a split needs to be performed
            split = tree.update(action, reward, current_state, new_state, episode_done, alpha, gamma, d, random_action,
                                hist_min_size)
            if split:
                tree.split_node(current_state, L)
                split_count += 1

            episode_done = episode_finished(new_state)

            if episode_done:
                iters_per_episode.append(j)
                avg_reward_per_episode.append(total_episode_reward / j)
                leaf_count.append(tree.get_total_leaves())
                node_count.append(tree.get_total_nodes())
            if step == num_of_steps:
                break
        if step == num_of_steps:
            break

    g = Digraph('G', filename='graph.gv')  # plot the learned tree in graphviz format
    tree.plot(g)
    print(g.source)
    print(f'iters per episode: {str(iters_per_episode)}')
    print(f'avg reward per episode: {avg_reward_per_episode}')
    print(f'total leaves per episode: {leaf_count}')
    print(f'total nodes per episode: {node_count}')
    print(f'tot. leaves/pos. leaves: {tree.get_total_leaves()}/{total_possible_leaves(lows, highs)}')
    print(f'tot. nodes: {tree.get_total_nodes()}')

    it_main = iters_per_episode
    rew_main = avg_reward_per_episode
    lc_main = leaf_count
    nc_main = node_count

    # uncomment for testing
    # it, rew, lc, nc = evaluate(model_path, tree, initial_state, D, gamma, alpha, d, num_of_episodes, num_of_steps)

    # reinitialize global variables
    tree.reinit_nodes()
    tree.reinit_leaves()
    tree.reinit_ids()

    # uncomment for testing
    return it_main, rew_main, lc_main, nc_main  # , it, rew, lc, nc


def take_action_old(current_state, epsilon, tree, step, num_of_steps):
    """
    Modified take action to suit the old algorithm's description. Main change:
    additionally return whether the random action was taken at step :param step.
    """
    eps_func = (lambda step: max(0.05, 1 - step / (num_of_steps)))
    if np.random.random() < eps_func(step):
        random_action = True  # True if action was chosen randomly
        action = random.choice(current_state.transitions)
        action_label = action.action.action_type.label
    else:
        random_action = False
        # select based on largest Q-Value
        action_label = tree.select_action(current_state)
        action = find_action_by_label(current_state, action_label)

    new_state = action.destinations.pick().state
    reward = get_immediate_reward(current_state, new_state)
    return action_label, reward, new_state, random_action


"""
Some of the useful runs, inc. the ones we used for our simulations. Don't forget to uncomment the evaluation
part of code at the end of both algorithms' definitions if evaluation is desired.
"""

# CQI("../Testing/models/resources_parsed_fully.jani")
# Old_Alg("../Testing/models/resources_parsed_fully.jani")

# save_full_stats_res_gath("../testing/Simulations/Sim_new_1000000steps.txt", 10, CQI,
# "../Testing/models/resources_parsed_fully.jani") save_full_stats_res_gath(
# "../testing/Simulations/Sim_old_1000000steps.txt", 10, Old_Alg, "../Testing/models/resources_parsed_fully.jani")

# hs_sim_res_gath("../testing/Simulations/hs_stats.txt", CQI, "../Testing/models/resources_parsed_fully.jani")

# CQI("../Testing/models/resources_parsed_fully.jani",
#     epsilon=0.5,
#     H_s=1,
#     D=0.999,
#     gamma=0.99,
#     alpha=0.1,
#     d=0.999,
#     num_of_episodes=5000,
#     num_of_steps=50000)

# Old_Alg("../Testing/models/resources_parsed_fully.jani",
#     epsilon=0.5,
#     gamma=0.99,
#     alpha=0.1,
#     d=0.999,
#     hist_min_size=200,
#     num_of_episodes=300,
#     num_of_steps = 50000)


"""
When running the Frozen lake model don't forget to:

1. Change the import of reward appropriately (lines 12-13)
2. Uncomment the lines:
    # lows.pop(0)
    # highs.pop(0)
    # var_labels.pop(0)
 in CQI/Old_alg functions
"""

# CQI("../Testing/models/lake.jani",
#     num_of_episodes=1000,
#     num_of_steps=100000)

# Old_Alg("../Testing/models/lake.jani",
#     num_of_episodes=1000,
#     num_of_steps=100000)
