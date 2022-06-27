from abc import abstractmethod

import numpy as np


class DecisionTree:
    def __init__(self):

        splits = [] #TODO: generate all possible splits by looking
                    # at the lowest and highest values at state space
        self.root = LeafNode(splits)

    def select_action(self, state):
        #select action with the largest Q value
        return np.argmax(self.root.get_qs(state))



class TreeNode:
    @abstractmethod
    def __init__(self, visit_decay):
        pass

    @abstractmethod
    def is_leaf(self):
        pass

    def select_action(self, state):
        #selects action with highest expected Q value
        pass


class LeafNode(TreeNode):

    def __init__(self, qs, feature, splits):
        self.qs = qs
        self.v = 0
        self.feature = feature
        self.splits = []
        self.splits = splits

    def is_leaf(self):
        return True

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

    def get_qs(self, state):
        return self.qs

    def select_child(self, state):
        if state[self.feature] < self.value:
            return self.left_child, self.right_child
        else:
            return self.right_child, self.left_child

class Inner_Node(TreeNode):
    #"branching node" in the paper

    def __init__(self, feature, value, left_child, right_child, visits):
        self.feature = feature
        self.value = value
        self.left_child = left_child
        self.right_child = right_child
        self.visits = visits

    def get_qs(self, s):
        # returns Q values of the corresponding child
        return self.select_child(s)[0].get_qs(s)