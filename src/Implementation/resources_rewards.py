REW_GOLD = 1000
REW_GEM = 1000
REW_ATTACKED = -1000


def get_immediate_reward(old_state, new_state):
    # defining our own immediate rewards for the resource gathering model

    required_gold_old = old_state.global_env["required_gold"].as_int
    required_gem_old = old_state.global_env["required_gem"].as_int

    attacked_new = new_state.global_env["attacked"].as_bool
    required_gold_new = new_state.global_env["required_gold"].as_int
    required_gem_new = new_state.global_env["required_gem"].as_int

    if required_gold_old - required_gold_new == 1:
        # brought gold home
        return REW_GOLD
    if required_gem_old - required_gem_new == 1:
        # brought gold home
        return REW_GEM

    if attacked_new == 1:
        # brought gem home
        return REW_ATTACKED

    return 0
