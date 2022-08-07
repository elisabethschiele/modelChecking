import operator
from abc import abstractmethod
import graphviz
from scipy import stats
import math

import numpy as np

# from src.Implementation.main import find_action_by_label
from graphviz import Digraph

node_counter = 0


class DecisionTreeOld:
    def __init__(self, initial_state, lows, highs, action_names):

        self.initial_state = initial_state
        self.lows = lows
        self.highs = highs
        self.action_names = action_names

        splits = self.generate_splits(lows, highs, action_names)
        for split in splits:
            print(str(split))

        actions_qs = {} # mapping of actions to Q-values
        for action in action_names:
            actions_qs[action] = 0
        self.root = LeafNode(actions_qs, 1, splits)
        # self.plot(self.g)
        # print(self.g.source)


    def select_action(self, state):
        # select action with highest q value among those that are allowed
        print(f'selecting best action for state {state.global_env}')
        print(f'action values: {self.root.get_qs(state).items()}')
        sorted_by_q = dict(sorted(self.root.get_qs(state).items(), key=lambda item: item[1], reverse=True))
        for key in sorted_by_q:
            if self.find_action_by_label(state, key) != -1:
                return key
        # return max(self.root.get_qs(state).items(), key=operator.itemgetter(1))[0]

    def find_action_by_label(self, state, label):
        for action in state.transitions:
            if action.action.action_type.label == label:
                return action
        print("No action found matching label " + label)
        return -1

    def generate_splits(self, lows, highs, action_names):

        default_qs = {}  # mapping of actions to Q-values
        for action in action_names:
            default_qs[action] = 0

        splits = []
        for f in range(len(highs)): # iterate for all features
            for i in range(highs[f]-lows[f]): # we want splits at 0.5, 1.5, 2.5, 3.5, ... -> i+0.5+lows[f]
                right_qs = {}  # mapping of actions to Q-values
                for action in action_names:
                    right_qs[action] = 0
                left_qs = {}  # mapping of actions to Q-values
                for action in action_names:
                    left_qs[action] = 0
                splits.append(Split(f,
                                    lows[f] + 0.5 + i,
                                    # np.zeros(total_actions),
                                    # np.zeros(total_actions),
                                    right_qs,
                                    left_qs,
                                    0.5,
                                    0.5))

        # generate two splits per feature: one at 1/3 distance between min and max and one at 2/3
        # for f in range(len(lows)):
        #     for i in range(4):
        #         splits.append(Split(f,
        #                             lows[f] + (highs[f] - lows[f]) / 5 * (i + 1),
        #                             np.zeros(total_actions),
        #                             np.zeros(total_actions),
        #                             0.5,
        #                             0.5))
        return splits

    def update(self, action, reward, old_state, new_state, episode_done, alpha, gamma, d, random_action, hist_min_size):
        # target = r + gamma * max(Q(new_state, best_next_action))
        target = reward + (gamma * max(self.root.get_qs(new_state).items(), key=operator.itemgetter(1))[1])

        return self.root.update(action, old_state, new_state, target, alpha, gamma, d, random_action, hist_min_size)

    def split_node(self, old_state, L):
        left_splits = self.generate_splits(self.lows, self.highs, self.action_names)
        right_splits = self.generate_splits(self.lows, self.highs, self.action_names)
        self.root = self.root.split_node(old_state, L, left_splits, right_splits)

    def best_split(self, state, action):
        # calls best_split on leaf corresponding to state
        L = self.root.get_leaf(state)
        return L.best_split(self, state, action)

    def structure(self):
        return self.root.structure()

    def plot(self, g):
        self.root.plot(g)

class Split():
    def __init__(self, feature, value, left_qs, right_qs, left_visits, right_visits):
        self.feature = feature # index of state variable
        self.value = value
        self.left_qs = left_qs
        self.right_qs = right_qs
        self.left_visits = left_visits
        self.right_visits = right_visits

    def __str__(self):
        return f"feature: {get_feature_name(self.feature)}, value: {self.value}, left_qs: {self.left_qs}, right_qs: {self.right_qs}, left_visits: {self.left_visits}, right_visits {self.right_visits}"



    def update(self, action, old_state, target, alpha, gamma, d):
        # feature 0 corresponds to x, feature 1 - to y
        # TODO: generalize - DONE
        feature_label = get_feature_name(self.feature)

        if old_state.global_env[feature_label].as_int < self.value:
            self.left_qs[action] = (1 - alpha) * self.left_qs[action] + alpha * target
            self.left_visits = self.left_visits * d + (1 - d)
            self.right_visits = self.right_visits * d
        else:
            self.right_qs[action] = (1 - alpha) * self.right_qs[action] + alpha * target
            self.right_visits = self.right_visits * d + (1 - d)
            self.left_visits = self.left_visits * d

        # print(str(self))

class TreeNode:
    #abstract class which might be deleted later
    @abstractmethod
    def __init__(self, visits):
        self.visits = visits

    @abstractmethod
    def is_leaf(self):
        pass

    @abstractmethod
    def get_qs(self, state):
        pass

    @abstractmethod
    def get_leaf(self, state):
        pass

    @abstractmethod
    def get_vs(self, state):
        pass

    # def update(self, action, old_state, target, alpha, gamma, d, random_action):
    #     # update visit frequency on the path
    #     self.visits = self.visits * d + (1 - d)
    #
    # def update_sibling(self, action, old_state, target, alpha, gamma, d):
    #     # update visit frequency of siblings on the path
    #     self.visits = self.visits * d

    @abstractmethod
    def split_node(self, old_state, L, left_splits, right_splits):
        pass

class LeafNode(TreeNode):

    visited_states =[]
    delta_q_hist = []

    def __init__(self, actions_qs, visits, splits):
        self.actions_qs = actions_qs
        self.visits = visits
        self.splits = splits
        global node_counter
        self.id = node_counter
        self.visited_states = []
        self.delta_q_hist = []
        node_counter += 1
        # print(f"id {self.id}, delta_q_hist {self.delta_q_hist}")
        # print(str(self))
        # print(self.structure())

    def __str__(self):
        splits_str = "\n"
        for split in self.splits:
            splits_str +=str(split)+"\n"
        return f"LNode. actions_qs: {self.actions_qs}, visits: {self.visits}, splits: {splits_str}"

    def structure(self):
        best_action = max(self.actions_qs, key=self.actions_qs.get)
        # print(self.actions_qs)
        # print(f"best_action: {best_action}")
        return f"best_action: {best_action}"
        # return str(self.actions_qs)

    def plot(self,g):
        g.node(str(self.id), max(self.actions_qs, key=self.actions_qs.get))

    def is_leaf(self):
        return True

    def get_qs(self, state):
        return self.actions_qs

    def update(self, action, old_state, new_state, target, alpha, gamma, d, random_action, hist_min_size):
        # update visit frequency
        # super().update(action, old_state, target, alpha, gamma, d, random_action)
        # print("visits: "+str(self.visits))

        # update leaf Q values
        # print("action_val: "+str(self.actions_qs[action]))

        # target = reward + (gamma * max(self.actions_qs.items(), key=operator.itemgetter(1))[1])
        print(f'target: {target}')

        updated_q_value = (1 - alpha) * self.actions_qs[action] + alpha * target
        delta_q = updated_q_value - self.actions_qs[action]
        print(f'old q value for {action}: {self.actions_qs[action]}, new: {updated_q_value}, q_vals: {self.actions_qs}')
        self.actions_qs[action] = updated_q_value

        split = False

        if not random_action:
        # if True:
            # add delta(Q) to the history list
            self.delta_q_hist.append(delta_q)
            self.visited_states.append(old_state)
            print(f'{old_state.global_env}, delta_q: {delta_q}')
            print(f'hist_list for id {self.id}: {self.delta_q_hist}')
            if len(self.delta_q_hist) >= hist_min_size:
                avg = sum(self.delta_q_hist) / len (self.delta_q_hist)
                std_dev = np.std(self.delta_q_hist)
                print(f'avg: {avg}, std_dev: {std_dev}')
                if abs(avg) < 2 * std_dev:
                    split = True


        # print("updated q_value of action \"" +str(action)+"\" is "+ str(self.actions_qs[action]))

        # update possible splits
        for s in self.splits:
            s.update(action, old_state, target, alpha, gamma, d)

        return split

    def get_leaf(self, state):
        return self

    def split_node(self, old_state, L, left_splits, right_splits):

        # determine the split with the highest T statistic
        print("VISITED STATES")
        for i in range(len(self.visited_states)):
            print(f'{self.visited_states[i].global_env}, delta_q: {self.delta_q_hist[i]}')
        max_t = 0
        best_split = None
        for split in self.splits:
            feat = get_feature_name(split.feature)
            val = split.value
            delta_q_left = []
            delta_q_right = []
            for i in range(len(self.visited_states)):
                # print(f'split_val: {val}, state_val {get_value(self.visited_states[i], feat)}, state: {self.visited_states[i].global_env}')
                if get_value(self.visited_states[i], feat) < val:
                    delta_q_left.append(self.delta_q_hist[i])
                else:
                    delta_q_right.append(self.delta_q_hist[i])

            t_stat = stats.ttest_ind(delta_q_left, delta_q_right)
            print("t_stat "+str(t_stat.statistic) + str(math.isnan(t_stat.statistic)))
            # TODO: do we need abs() and not NaN here?
            if abs(t_stat.statistic) > max_t and not math.isnan(t_stat.statistic):
                best_split = split
                max_t = abs(t_stat.statistic)

            # JUST COMPARE AVG Q VALUES
            # if len(delta_q_left) != 0 and len(delta_q_right) != 0:
            #     avg_q_left = sum(delta_q_left) / len (delta_q_left)
            #     avg_q_right = sum(delta_q_right) / len(delta_q_right)
            #     curr_t = abs(avg_q_left - avg_q_right)
            #     print(f'diff_in_q: {curr_t}')
            #     if curr_t > max_t:
            #         best_split = split
            #         max_t = curr_t
            print(f'feat {feat}, val {val}')
            print(f'delta_q_left: {delta_q_left}')
            print(f'delta_q_right: {delta_q_right}')
        if best_split is None:
            return self

        print(f'best split: {str(best_split)}')

        Bv = self.visits
        Bu = best_split.feature
        Bm = best_split.value

        Lv = best_split.left_visits
        Lq = best_split.left_qs
        Rv = best_split.right_visits
        Rq = best_split.right_qs

        Left_Child = LeafNode(Lq, Lv, left_splits)
        Right_Child = LeafNode(Rq, Rv, right_splits)
        B = Inner_Node(Bu, Bm, Left_Child, Right_Child, Bv)
        return B

    def best_split(self, Tree, state, action):
        Vp = Tree.root.get_vs(state)
        # print(f'vp = {Vp}')
        SQ = []
        for i in range(len(self.splits)):
            split = self.splits[i]
            cl_array = []
            for key in split.left_qs:
                cl_array.append(split.left_qs[key] - self.actions_qs[action])
            # print(f'all cl values: {cl_array}')
            # print(f'all left qs are: {split.left_qs}')
            # print(f'action is: {action}')
            # print(f'all action qs for current state are: {self.actions_qs}')
            # print(f'left q for a\'{split.left_qs[key]}')
            # print(f'current q for {action}: {self.actions_qs[action]}')
            cl = max(cl_array)
            cr_array = []
            for key in split.right_qs:
                cr_array.append(split.right_qs[key] - self.actions_qs[action])
            cr = max(cr_array)
            SQ.append(Vp * (cl * split.left_visits + cr * split.right_visits))
            # print(f"i = {i}")
            # print(f'length of SQ = {len(SQ)}')

            # print(f'{Vp} * ({cl} * {split.left_visits} + {cr} * {split.right_visits}) = {SQ[i]}')
        print(f'SQ = {SQ}')
        bestSplit = self.splits[np.argmax(SQ)]
        # print(f"bestSplit: {bestSplit}")
        bestValue = max(SQ)
        return bestSplit, bestValue

    def get_vs(self, state):
        # returns values of all visits from root to leaf corresponding to state
        return self.visits






class Inner_Node(TreeNode):
    #"branching node" in the paper

    def __init__(self, feature, value, left_child, right_child, visits):
        self.feature = feature
        self.value = value
        self.left_child = left_child
        self.right_child = right_child
        self.visits = visits
        global node_counter
        self.id = node_counter
        node_counter += 1
        # print(str(self))

    def __str__(self):
        return f"BNode. feature: {self.feature}, value: {self.value}, visits: {self.visits}"

    def structure(self):
        return "{"+str(self.feature)+": "+str(self.value)+" "+self.left_child.structure()+","+self.right_child.structure()+"}"

    def plot(self, g):
        self.left_child.plot(g)
        self.right_child.plot(g)
        g.node(str(self.id), f"{get_feature_name(self.feature)}:{self.value}")
        g.edge(str(self.id), str(self.left_child.id), label = "<")
        g.edge(str(self.id), str(self.right_child.id), label = ">")


    def is_leaf(self):
        return False

    def get_qs(self, s):
        # returns Q values of the corresponding child
        return self.select_child(s)[0].get_qs(s)

    def select_child(self, state):
        # selects the child that corresponds to the state
        # TODO: generalize - Done
        feature_label = get_feature_name(self.feature)

        if state.global_env[feature_label].as_int < self.value:
            return self.left_child, self.right_child
        else:
            return self.right_child, self.left_child

    def get_leaf(self, state):
        return self.select_child(state)[0].get_leaf(state)

    def update(self, action, old_state, new_state, target, alpha, gamma, d, random_action, hist_min_size):
        #TODO

        # update visit frequency
        # super().update(action, old_state, target, alpha, gamma, d)

        # continue updating in the direction of respective leaf
        next_node, sibling = self.select_child(old_state)
        return next_node.update(action, old_state, new_state, target, alpha, gamma, d, random_action, hist_min_size)
        # sibling.update_sibling(action, old_state, target, alpha, gamma, d)

    def split_node(self, old_state, L, left_splits, right_splits):
        # TODO: generalize - Done
        feature_label = get_feature_name(self.feature)

        if old_state.global_env[feature_label].as_int < self.value:
            self.left_child = self.left_child.split_node(old_state, L, left_splits, right_splits)
        else:
            self.right_child = self.right_child.split_node(old_state, L, left_splits, right_splits)
        return self

    def get_vs(self, state):
        # returns values of all visits from root to leaf corresponding to state
        return self.visits * self.select_child(state)[0].get_vs(state)

def get_feature_name(index):
        if index == 0:
            return "x"
        if index == 1:
            return "y"
        if index == 2:
            return "required_gold"
        if index == 3:
            return "required_gem"
        if index == 4:
            return "gold"
        if index == 5:
            return "gem"
        if index == 6:
            return "attacked"
        return "none"

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
