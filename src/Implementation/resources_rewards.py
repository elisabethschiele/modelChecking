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
        # brought gem home
        return REW_GEM
    if attacked_new == 1:
        # was attacked
        return REW_ATTACKED
    return 0


def episode_finished(state):
    # episode is done as soon as no more gold or gems are required
    return get_value(state, "required_gold") == 0 and get_value(state, "required_gem") == 0


def get_value(state, variable_name):
    """
    Function is not relevant anymore, but might be useful for further development.
    Returns integer value of :param variable_name of state :param state.
    """
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
