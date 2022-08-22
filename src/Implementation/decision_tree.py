from abc import abstractmethod


class DecisionTree:
    """
    This is the generic tree class that includes functionality common for
    the decision trees for each of two algorithms (see decision_tree_new.py
    and decision_tree_old.py)
    """

    def __init__(self, initial_state, lows, highs, action_names, var_labels):

        self.initial_state = initial_state
        self.lows = lows
        self.highs = highs
        self.action_names = action_names
        self.var_labels = var_labels
        self.splits = self.generate_splits(lows, highs, action_names)

        self.actions_qs = {}  # mapping of actions to Q-values
        for action in action_names:
            self.actions_qs[action] = 0

    def select_action(self, state):
        # select action with highest q value among those that are allowed
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
        for f in range(len(highs)):  # iterate over all features
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
                                    0.5, self.var_labels))
        return splits

    def structure(self):
        return self.root.structure()

    def plot(self, g):
        self.root.plot(g)


class Split():
    def __init__(self, feature, value, left_qs, right_qs, left_visits, right_visits, var_labels):
        self.feature = feature  # index of state variable
        self.value = value
        self.left_qs = left_qs
        self.right_qs = right_qs
        self.left_visits = left_visits
        self.right_visits = right_visits
        self.var_labels = var_labels

    def __str__(self):
        return f"feature: {self.var_labels[self.feature]}, value: {self.value}, left_qs: {self.left_qs}, right_qs: {self.right_qs}, left_visits: {self.left_visits}, right_visits {self.right_visits}"

    def update(self, action, old_state, target, alpha, gamma, d):
        # feature 0 corresponds to x, feature 1 - to y
        feature_label = self.var_labels[self.feature]

        if old_state.global_env[feature_label].as_int < self.value:
            self.left_qs[action] = (1 - alpha) * self.left_qs[action] + alpha * target
            self.left_visits = self.left_visits * d + (1 - d)
            self.right_visits = self.right_visits * d
        else:
            self.right_qs[action] = (1 - alpha) * self.right_qs[action] + alpha * target
            self.right_visits = self.right_visits * d + (1 - d)
            self.left_visits = self.left_visits * d


class TreeNode:
    # abstract class to keep a better structure
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

    def update_sibling(self, action, old_state, target, alpha, gamma, d):
        # update visit frequency of siblings on the path
        self.visits = self.visits * d


class LeafNode(TreeNode):
    # generic leaf node
    def __init__(self, actions_qs, visits, splits, var_labels):
        self.actions_qs = actions_qs
        self.visits = visits
        self.splits = splits
        self.var_labels = var_labels

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

    def get_leaf(self, state):
        return self

    def get_vs(self, state):
        # returns values of all visits from root to leaf corresponding to state
        return self.visits


class InnerNode(TreeNode):
    # "branching node" in the paper, once again, generic: each of the algorthims
    # uses an extension of this class

    def __init__(self, feature, value, left_child, right_child, visits, var_labels):
        self.feature = feature
        self.value = value
        self.left_child = left_child
        self.right_child = right_child
        self.visits = visits
        # global id_counter
        # self.id = id_counter
        # id_counter += 1
        self.var_labels = var_labels

    def __str__(self):
        return f"BNode. feature: {self.feature}, value: {self.value}, visits: {self.visits}"

    def structure(self):
        return "{" + str(self.feature) + ": " + str(
            self.value) + " " + self.left_child.structure() + "," + self.right_child.structure() + "}"

    def plot(self, g):
        self.left_child.plot(g)
        self.right_child.plot(g)
        g.node(str(self.id), f"{self.var_labels[self.feature]}:{self.value}")
        g.edge(str(self.id), str(self.left_child.id), label="<")
        g.edge(str(self.id), str(self.right_child.id), label=">")

    def is_leaf(self):
        return False

    def get_qs(self, s):
        # returns Q values of the corresponding child
        return self.select_child(s)[0].get_qs(s)

    def get_leaf(self, state):
        return self.select_child(state)[0].get_leaf(state)

    def get_vs(self, state):
        # return values of all visits from root to leaf corresponding to state
        return self.visits * self.select_child(state)[0].get_vs(state)

    def select_child(self, state):
        # selects the child that corresponds to the state
        feature_label = self.var_labels[self.feature]
        if state.global_env[feature_label].as_int < self.value:
            return self.left_child, self.right_child
        else:
            return self.right_child, self.left_child
