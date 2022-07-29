import operator
from abc import abstractmethod

import numpy as np


class DecisionTree:
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

    def select_action(self, state):
        # select action with the largest Q value
        return max(self.root.get_qs(state).items(), key=operator.itemgetter(1))[0]
        # return np.argmax(self.root.get_qs(state))

    def generate_splits(self, lows, highs, action_names):

        default_qs = {}  # mapping of actions to Q-values
        for action in action_names:
            default_qs[action] = 0

        splits = []
        for f in range(2): # two features we split upon: x and y
            for i in range(4): # we want 4 splits: at 1.5, 2.5, 3.5, 4.5
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

    def update(self, action, reward, old_state, new_state, episode_done, alpha, gamma, d):
        # TODO: reduce params to necessary
        # target = r + gamma * max(Q(new_state, best_next_action))
        if episode_done:
            target = reward
        else:
            target = reward + (gamma * max(self.root.get_qs(new_state).items(), key=operator.itemgetter(1))[1])

        print("target: "+str(target))

        self.root.update(action, old_state, target, alpha, gamma, d)

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

    def update(self, action, old_state, target, alpha, gamma, d):
        # feature 0 corresponds to x, feature 1 - to y
        feature_label = "x" if self.feature == 0 else "y"

        if old_state.global_env[feature_label].as_int < self.value:
            self.left_qs[action] = (1 - alpha) * self.left_qs[action] + alpha * target
            self.left_visits = self.left_visits * d + (1 - d)
            self.right_visits = self.right_visits * d
        else:
            self.right_qs[action] = (1 - alpha) * self.right_qs[action] + alpha * target
            self.right_visits = self.right_visits * d + (1 - d)
            self.left_visits = self.left_visits * d

        print(str(self))

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

    def update(self, action, old_state, target, alpha, gamma, d):
        # update visit frequency on the path
        self.visits = self.visits * d + (1 - d)

    def update_sibling(self, action, old_state, target, alpha, gamma, d):
        # update visit frequency of siblings on the path
        self.visits = self.visits * d

class LeafNode(TreeNode):

    def __init__(self, actions_qs, visits, splits):
        self.actions_qs = actions_qs
        self.visits = visits
        self.splits = splits

    def is_leaf(self):
        return True

    def get_qs(self, state):
        return self.actions_qs

    def update(self, action, old_state, target, alpha, gamma, d):
        # update visit frequency
        super().update(action, old_state, target, alpha, gamma, d)
        print("visits: "+str(self.visits))

        # update leaf Q values
        print("action_val: "+str(self.actions_qs[action]))
        self.actions_qs[action] = (1 - alpha) * self.actions_qs[action] + alpha * target
        print("updated q_value of action \"" +str(action)+"\" is "+ str(self.actions_qs[action]))

        # update possible splits
        for split in self.splits:
            split.update(action, old_state, target, alpha, gamma, d)

    def split(self, best_split):
        #Algorithm 7
        Bv = self.visits
        Bu = best_split.feature
        
        Lv = best_split.left_visits
        Lq = best_split.left_qs
        Rv = best_split.right_visits
        Rq = best_split.right_qs

        L = LeafNode(Lq, Lv, self.splits)
        R = LeafNode(Rq, Rv, self.splits)
        B = Inner_Node(Bu, Bv, L, R, self.visits)
        return B, L, R

    def get_leaf(self, state):
        return self

    def best_split(self, Tree, state, action):
        # TODO
        Vp = Tree.root.get_vs(state)
        pass

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

    def update(self, action, old_state, target, alpha, gamma, d):
        # update visit frequency
        super().update(action, old_state, target, alpha, gamma, d)

        # continue updating in the direction of respective leaf
        next_node, sibling = self.select_child(old_state)
        next_node.update(action, old_state, target, alpha, gamma, d)
        sibling.update_sibling(action, old_state, target, alpha, gamma, d)

    def get_vs(self, state):
        # returns values of all visits from root to leaf corresponding to state
        return self.visits * self.select_child(state)[0].get_vs(state)