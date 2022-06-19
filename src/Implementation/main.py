from momba import engine, jani
import gym
import pathlib
import decision_tree

#path = pathlib.Path("../Testing/models/firewire.true.jani")
# path = pathlib.Path("../Testing/models/die.jani")
# network = jani.load_model(path.read_text("utf-8"))
#
# explorer = engine.Explorer.new_discrete_time(
#     network,
#     parameters={
#         "delay": 3,
#         "deadline": 200,
#     }
# )
#
# initial_states = explorer.initial_states
# (initial_state,) = initial_states
# print(initial_state.global_env)

def CQI(network, H_s = 0.5, D = 0.5, num_of_episodes=10):
    tree = decision_tree.DecisionTree()
    #Algorithm 1


