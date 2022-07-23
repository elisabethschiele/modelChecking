REW_GOLD = 20
REW_GEM = 20


def get_immediate_reward(old_state, new_state):
    # defining our own immediate rewards for the resource gathering model

    # x_old = old_state.global_env["x"].as_int
    # y_old = old_state.global_env["y"].as_int
    gold_old = old_state.global_env["gold"].as_bool
    gem_old = old_state.global_env["gem"].as_bool
    # attacked_old = old_state.global_env["attacked"].as_bool
    # required_gold_old = old_state.global_env["required_gold"].as_int
    # required_gem_old = old_state.global_env["required_gem"].as_int

    x_new = new_state.global_env["x"].as_int
    y_new = new_state.global_env["y"].as_int
    # gold_new = new_state.global_env["gold"].as_bool
    # gem_new = new_state.global_env["gem"].as_bool
    # attacked_new = new_state.global_env["attacked"].as_bool
    # required_gold_new = new_state.global_env["required_gold"].as_int
    # required_gem_new = new_state.global_env["required_gem"].as_int

    if gold_old and x_new == 3 and y_new == 1:
        # brought gold home
        return REW_GOLD
    if gem_old and x_new == 3 and y_new == 1:
        # brought gold home
        return REW_GEM

    return -1
