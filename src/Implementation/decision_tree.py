import operator
from abc import abstractmethod

import numpy as np


class DecisionTree:
    def __init__(self, initial_state, lows, highs, action_names):

        self.initial_state = initial_state
        self.lows = lows
        self.highs = highs
        self.action_names = action_names

        splits = self.generate_splits(lows, highs, len(action_names))
        for split in splits:
            print(str(split))

        actions_qs = {} # mapping of actions to Q-values
        for action in action_names:
            actions_qs[action] = 0
        self.root = LeafNode(actions_qs, 1, splits)

    def select_action(self, state):
        #select action with the largest Q value
        return max(self.root.get_qs(state).items(), key=operator.itemgetter(1))[0]
        # return np.argmax(self.root.get_qs(state))

    def generate_splits(self, lows, highs, total_actions):

        splits = []
        for f in range(2): # two features we split upon: x and y
            for i in range(4): # we want 4 splits: at 1.5, 2.5, 3.5, 4.5
                splits.append(Split(f,
                                    lows[f] + 0.5 + i,
                                    np.zeros(total_actions),
                                    np.zeros(total_actions),
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

    def update(self, L, action, reward, old_state, new_state, episode_done, gamma):
        # target = r + gamma * max(Q(new_state, best_next_action))
        if episode_done:
            target = reward
        else:
            target = reward + (gamma * np.max(self.root.get_qs(new_state)))
        # self.root.update(s, a, target, self.params)

class Split():
    def __init__(self, feature, value, left_qs, right_qs, left_visits, right_visits):
        self.feature = feature # index of state variable
        self.value = value
        self.left_qs = left_qs
        self.right_qs = right_qs
        self.left_visits = left_visits
        self.right_visits = right_visits

    def __str__(self):
        return f"feature: {self.feature}, value: {self.value}, left_qs: {self.left_qs}, right_qs: {self.right_qs}, left_visits: {self.left_visits}, right_visits {self.right_visits}"



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

    def __init__(self, actions_qs, visits, splits):
        self.actions_qs = actions_qs
        self.visits = visits
        self.splits = splits

    def is_leaf(self):
        return True

    def get_qs(self, state):
        return self.actions_qs

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
