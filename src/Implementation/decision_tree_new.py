import operator
import numpy as np

from src.Implementation.decision_tree import DecisionTree
from src.Implementation.decision_tree import LeafNode
from src.Implementation.decision_tree import InnerNode

id_counter = 0
leaf_counter = 0
node_counter = 0


class DecisionTreeNew(DecisionTree):
    # Decision tree for the CQI algorithm

    def __init__(self, initial_state, lows, highs, action_names, var_labels):
        super().__init__(initial_state, lows, highs, action_names, var_labels)
        self.root = LeafNodeNew(self.actions_qs, 1, self.splits, self.var_labels)

    def select_action(self, state):
        return super().select_action(state)

    def find_action_by_label(self, state, label):
        return super().find_action_by_label(state, label)

    def generate_splits(self, lows, highs, action_names):
        return super().generate_splits(lows, highs, action_names)

    def update(self, action, reward, old_state, new_state, episode_done, alpha, gamma, d):
        target = reward + (gamma * max(self.root.get_qs(new_state).items(), key=operator.itemgetter(1))[1])
        self.root.update(action, old_state, target, alpha, gamma, d)

    def split_node(self, old_state, L, best_split):
        left_splits = self.generate_splits(self.lows, self.highs, self.action_names)
        right_splits = self.generate_splits(self.lows, self.highs, self.action_names)
        self.root = self.root.split_node(old_state, L, best_split, left_splits, right_splits)

    def best_split(self, state, action):
        # calls best_split on leaf corresponding to state
        L = self.root.get_leaf(state)
        return L.best_split(self, state, action)

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


class LeafNodeNew(LeafNode):
    def __init__(self, actions_qs, visits, splits, var_labels):
        super().__init__(actions_qs, visits, splits, var_labels)
        global id_counter
        self.id = id_counter
        id_counter += 1
        global leaf_counter
        leaf_counter += 1
        global node_counter
        node_counter += 1

    def update(self, action, old_state, target, alpha, gamma, d):
        # update visit frequency
        self.visits = self.visits * d + (1 - d)
        # update leaf Q values
        self.actions_qs[action] = (1 - alpha) * self.actions_qs[action] + alpha * target

        # update possible splits
        for split in self.splits:
            split.update(action, old_state, target, alpha, gamma, d)

    def split_node(self, old_state, L, best_split, left_splits, right_splits):
        # Algorithm 7
        Bv = self.visits
        Bu = best_split.feature
        Bm = best_split.value

        Lv = best_split.left_visits
        Lq = best_split.left_qs
        Rv = best_split.right_visits
        Rq = best_split.right_qs

        Left_Child = LeafNodeNew(Lq, Lv, left_splits, self.var_labels)
        Right_Child = LeafNodeNew(Rq, Rv, right_splits, self.var_labels)
        B = InnerNodeNew(Bu, Bm, Left_Child, Right_Child, Bv, self.var_labels)

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


class InnerNodeNew(InnerNode):

    def __init__(self, feature, value, left_child, right_child, visits, var_labels):
        super().__init__(feature, value, left_child, right_child, visits, var_labels)
        global id_counter
        self.id = id_counter
        id_counter += 1

    def select_child(self, state):
        # selects the child that corresponds to the state
        feature_label = self.var_labels[self.feature]
        if state.global_env[feature_label].as_int < self.value:
            return self.left_child, self.right_child
        else:
            return self.right_child, self.left_child

    def update(self, action, old_state, target, alpha, gamma, d):
        # update visit frequency
        self.visits = self.visits * d + (1 - d)
        # continue updating in the direction of respective leaf
        next_node, sibling = self.select_child(old_state)
        next_node.update(action, old_state, target, alpha, gamma, d)
        sibling.update_sibling(action, old_state, target, alpha, gamma, d)

    def split_node(self, old_state, L, best_split, left_splits, right_splits):
        feature_label = self.var_labels[self.feature]

        if old_state.global_env[feature_label].as_int < self.value:
            self.left_child = self.left_child.split_node(old_state, L, best_split, left_splits, right_splits)
        else:
            self.right_child = self.right_child.split_node(old_state, L, best_split, left_splits, right_splits)
        return self
