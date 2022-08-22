import operator
from scipy import stats
import math
import warnings
import numpy as np

from src.Implementation.decision_tree import DecisionTree
from src.Implementation.decision_tree import LeafNode
from src.Implementation.decision_tree import InnerNode

id_counter = 0
leaf_counter = 0
node_counter = 0

warnings.filterwarnings("ignore")  # ignore irrelevant precision loss warnings


class DecisionTreeOld(DecisionTree):
    # Decision tree for the Pyeatt algorithm

    def __init__(self, initial_state, lows, highs, action_names, var_labels):
        super().__init__(initial_state, lows, highs, action_names, var_labels)
        self.root = LeafNodeOld(self.actions_qs, 1, self.splits, self.var_labels)

    def select_action(self, state):
        return super().select_action(state)

    def find_action_by_label(self, state, label):
        return super().find_action_by_label(state, label)

    def generate_splits(self, lows, highs, action_names):
        return super().generate_splits(lows, highs, action_names)

    def update(self, action, reward, old_state, new_state, episode_done, alpha, gamma, d, random_action, hist_min_size):
        target = reward + (gamma * max(self.root.get_qs(new_state).items(), key=operator.itemgetter(1))[1])
        return self.root.update(action, old_state, new_state, target, alpha, gamma, d, random_action, hist_min_size)

    def split_node(self, old_state, L):
        left_splits = self.generate_splits(self.lows, self.highs, self.action_names)
        right_splits = self.generate_splits(self.lows, self.highs, self.action_names)
        self.root = self.root.split_node(old_state, L, left_splits, right_splits)

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


class LeafNodeOld(LeafNode):
    visited_states = []
    delta_q_hist = []

    def __init__(self, actions_qs, visits, splits, var_labels):
        super().__init__(actions_qs, visits, splits, var_labels)
        global id_counter
        self.id = id_counter
        self.visited_states = []
        self.delta_q_hist = []
        id_counter += 1
        global leaf_counter
        leaf_counter += 1
        global node_counter
        node_counter += 1

    def update(self, action, old_state, new_state, target, alpha, gamma, d, random_action, hist_min_size):

        updated_q_value = (1 - alpha) * self.actions_qs[action] + alpha * target
        delta_q = updated_q_value - self.actions_qs[action]
        self.actions_qs[action] = updated_q_value
        split = False

        if not random_action:
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

        Left_Child = LeafNodeOld(Lq, Lv, left_splits, self.var_labels)
        Right_Child = LeafNodeOld(Rq, Rv, right_splits, self.var_labels)
        B = InnerNodeOld(Bu, Bm, Left_Child, Right_Child, Bv, self.var_labels)

        global leaf_counter
        leaf_counter -= 1
        return B


class InnerNodeOld(InnerNode):

    def __init__(self, feature, value, left_child, right_child, visits, var_labels):
        super().__init__(feature, value, left_child, right_child, visits, var_labels)
        global id_counter
        self.id = id_counter
        id_counter += 1

    def select_child(self, state):
        # selects the child that corresponds to the state
        feature_label = get_feature_name(self.feature)
        if state.global_env[feature_label].as_int < self.value:
            return self.left_child, self.right_child
        else:
            return self.right_child, self.left_child

    def update(self, action, old_state, new_state, target, alpha, gamma, d, random_action, hist_min_size):
        # continue updating in the direction of respective leaf
        next_node, sibling = self.select_child(old_state)
        return next_node.update(action, old_state, new_state, target, alpha, gamma, d, random_action, hist_min_size)

    def split_node(self, old_state, L, left_splits, right_splits):
        feature_label = get_feature_name(self.feature)
        # feature_label = self.var_labels[self.feature]
        if old_state.global_env[feature_label].as_int < self.value:
            self.left_child = self.left_child.split_node(old_state, L, left_splits, right_splits)
        else:
            self.right_child = self.right_child.split_node(old_state, L, left_splits, right_splits)
        return self


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
