from abc import abstractmethod

import numpy as np


class DecisionTree:
    def __init__(self, initial_state, lows, highs, total_actions):

        self.initial_state = initial_state
        self.lows = lows
        self.highs = highs
        self.total_actions = total_actions

        splits = self.generate_splits(lows, highs, total_actions)
        self.root = LeafNode(np.zeros(total_actions), 1, splits)

    def select_action(self, state):
        #select action with the largest Q value
        return np.argmax(self.root.get_qs(state))

    def generate_splits(self, lows, highs, total_actions):
        # generate two splits per feature: one at 1/3 distance between min and max and one at 2/3
        splits = []
        for f in range(len(lows)):
            for i in range(2):
                splits.append(Split(f,
                                    lows[f] + (highs[f] - lows[f]) / 3 * (i + 1),
                                    np.zeros(total_actions),
                                    np.zeros(total_actions),
                                    0.5,
                                    0.5))
        return splits

class Split():
    def __init__(self, feature, value, left_qs, right_qs, left_visits, right_visits):
        self.feature = feature # index of state variable
        self.value = value
        self.left_qs = left_qs
        self.right_qs = right_qs
        self.left_visits = left_visits
        self.right_visits = right_visits



class TreeNode:
    #abstract class which might be deleted later
    @abstractmethod
    def __init__(self, visits):
        self.visits = visits
        pass

    @abstractmethod
    def is_leaf(self):
        pass

    @abstractmethod
    def get_qs(self, state):
        pass

    @abstractmethod
    def get_leaf(self, state):
        pass

class LeafNode(TreeNode):

    def __init__(self, qs, visits, splits):
        self.qs = qs
        self.visits = visits
        self.splits = splits

    def is_leaf(self):
        return True

    def get_qs(self, state):
        return self.qs

    def update_leaf_q_values(self):
        #Algorithm 3
        pass

    def update_visit_frequency(self):
        #Algorithm 4
        pass

    def update_possible_splits(self):
        #Algorithm 5
        for split in self.splits:
            split.update()

    def split(self, best_split):
        #Algorithm 7
        pass

    def get_leaf(self, state):
        return self





class Inner_Node(TreeNode):
    #"branching node" in the paper

    def __init__(self, feature, value, left_child, right_child, visits):
        self.feature = feature
        self.value = value
        self.left_child = left_child
        self.right_child = right_child
        self.visits = visits

    def is_leaf(self):
        return False

    def get_qs(self, s):
        # returns Q values of the corresponding child
        return self.select_child(s)[0].get_qs(s)

    def select_child(self, state):
        # selects the child that corresponds to the state
        if state[self.feature] < self.value:
            return self.left_child, self.right_child
        else:
            return self.right_child, self.left_child

    def get_leaf(self, state):
        return self.select_child(state)[0].get_leaf(state)
