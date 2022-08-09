import operator
from abc import abstractmethod
from scipy import stats
import math
import warnings
import numpy as np

id_counter = 0
leaf_counter = 0
node_counter = 0

warnings.filterwarnings("ignore") # ignore irrelevant precision loss warnings

# TODO: merge with decision_tree.py as mush as possible
# TODO: clean up unused methods, parameters

class DecisionTreeOld:
    def __init__(self, initial_state, lows, highs, action_names):

        self.initial_state = initial_state
        self.lows = lows
        self.highs = highs
        self.action_names = action_names
        splits = self.generate_splits(lows, highs, action_names)
        for split in splits:
            print(str(split))
        actions_qs = {}  # mapping of actions to Q-values
        for action in action_names:
            actions_qs[action] = 0
        self.root = LeafNode(actions_qs, 1, splits)

    def select_action(self, state):
        sorted_by_q = dict(sorted(self.root.get_qs(state).items(), key=lambda item: item[1], reverse=True))
        for key in sorted_by_q:
            if self.find_action_by_label(state, key) != -1:
                return key

    def find_action_by_label(self, state, label):
        for action in state.transitions:
            if action.action.action_type.label == label:
                return action
        return -1

    def generate_splits(self, lows, highs, action_names):

        default_qs = {}  # mapping of actions to Q-values
        for action in action_names:
            default_qs[action] = 0

        splits = []
        for f in range(len(highs)):  # iterate for all features
            for i in range(highs[f] - lows[f]):  # we want splits at 0.5, 1.5, 2.5, 3.5, ... -> i+0.5+lows[f]
                right_qs = {}  # mapping of actions to Q-values
                for action in action_names:
                    right_qs[action] = 0
                left_qs = {}  # mapping of actions to Q-values
                for action in action_names:
                    left_qs[action] = 0
                splits.append(Split(f,
                                    lows[f] + 0.5 + i,
                                    right_qs,
                                    left_qs,
                                    0.5,
                                    0.5))
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

    def get_total_leaves(self):
        global leaf_counter
        return leaf_counter

    def get_total_nodes(self):
        global node_counter
        return node_counter

    def reinit_leaves(self):
        global leaf_counter
        leaf_counter = 0

    def reinit_nodes(self):
        global node_counter
        node_counter = 0

    def reinit_ids(self):
        global id_counter
        id_counter = 0


class Split():
    def __init__(self, feature, value, left_qs, right_qs, left_visits, right_visits):
        self.feature = feature  # index of state variable
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


class TreeNode:
    # abstract class for better structure
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

    @abstractmethod
    def split_node(self, old_state, L, left_splits, right_splits):
        pass


class LeafNode(TreeNode):
    visited_states = []
    delta_q_hist = []

    def __init__(self, actions_qs, visits, splits):
        self.actions_qs = actions_qs
        self.visits = visits
        self.splits = splits
        global id_counter
        self.id = id_counter
        self.visited_states = []
        self.delta_q_hist = []
        id_counter += 1
        global leaf_counter
        leaf_counter += 1
        global node_counter
        node_counter += 1

    def __str__(self):
        splits_str = "\n"
        for split in self.splits:
            splits_str += str(split) + "\n"
        return f"LNode. actions_qs: {self.actions_qs}, visits: {self.visits}, splits: {splits_str}"

    def structure(self):
        best_action = max(self.actions_qs, key=self.actions_qs.get)
        return f"best_action: {best_action}"

    def plot(self, g):
        g.node(str(self.id), max(self.actions_qs, key=self.actions_qs.get))

    def is_leaf(self):
        return True

    def get_qs(self, state):
        return self.actions_qs

    def update(self, action, old_state, new_state, target, alpha, gamma, d, random_action, hist_min_size):

        updated_q_value = (1 - alpha) * self.actions_qs[action] + alpha * target
        delta_q = updated_q_value - self.actions_qs[action]
        self.actions_qs[action] = updated_q_value

        split = False

        if not random_action:
            # if True:
            # add delta(Q) to the history list
            self.delta_q_hist.append(delta_q)
            self.visited_states.append(old_state)
            if len(self.delta_q_hist) >= hist_min_size:
                avg = sum(self.delta_q_hist) / len(self.delta_q_hist)
                std_dev = np.std(self.delta_q_hist)
                if abs(avg) < 2 * std_dev:
                    split = True
        # update possible splits
        for s in self.splits:
            s.update(action, old_state, target, alpha, gamma, d)

        return split

    def get_leaf(self, state):
        return self

    def split_node(self, old_state, L, left_splits, right_splits):

        # determine the split with the highest T statistic
        max_t = 0
        best_split = None
        for split in self.splits:
            feat = get_feature_name(split.feature)
            val = split.value
            delta_q_left = []
            delta_q_right = []
            for i in range(len(self.visited_states)):
                if get_value(self.visited_states[i], feat) < val:
                    delta_q_left.append(self.delta_q_hist[i])
                else:
                    delta_q_right.append(self.delta_q_hist[i])
            t_stat = stats.ttest_ind(delta_q_left, delta_q_right)

            if abs(t_stat.statistic) > max_t and not math.isnan(t_stat.statistic):
                best_split = split
                max_t = abs(t_stat.statistic)

        if best_split is None:
            return self

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

        global leaf_counter
        leaf_counter -= 1
        return B

    def best_split(self, Tree, state, action):
        Vp = Tree.root.get_vs(state)
        SQ = []
        for i in range(len(self.splits)):
            split = self.splits[i]
            cl_array = []
            for key in split.left_qs:
                cl_array.append(split.left_qs[key] - self.actions_qs[action])
            cl = max(cl_array)
            cr_array = []
            for key in split.right_qs:
                cr_array.append(split.right_qs[key] - self.actions_qs[action])
            cr = max(cr_array)
            SQ.append(Vp * (cl * split.left_visits + cr * split.right_visits))
        bestSplit = self.splits[np.argmax(SQ)]
        bestValue = max(SQ)
        return bestSplit, bestValue

    def get_vs(self, state):
        # returns values of all visits from root to leaf corresponding to state
        return self.visits


class Inner_Node(TreeNode):
    # "branching node" in the paper

    def __init__(self, feature, value, left_child, right_child, visits):
        self.feature = feature
        self.value = value
        self.left_child = left_child
        self.right_child = right_child
        self.visits = visits
        global id_counter
        self.id = id_counter
        id_counter += 1

    def __str__(self):
        return f"BNode. feature: {self.feature}, value: {self.value}, visits: {self.visits}"

    def structure(self):
        return "{" + str(self.feature) + ": " + str(
            self.value) + " " + self.left_child.structure() + "," + self.right_child.structure() + "}"

    def plot(self, g):
        self.left_child.plot(g)
        self.right_child.plot(g)
        g.node(str(self.id), f"{get_feature_name(self.feature)}:{self.value}")
        g.edge(str(self.id), str(self.left_child.id), label="<")
        g.edge(str(self.id), str(self.right_child.id), label=">")

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

        # continue updating in the direction of respective leaf
        next_node, sibling = self.select_child(old_state)
        return next_node.update(action, old_state, new_state, target, alpha, gamma, d, random_action, hist_min_size)

    def split_node(self, old_state, L, left_splits, right_splits):
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
