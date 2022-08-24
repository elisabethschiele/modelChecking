REW_GOAL = 100
REW_HOLE = -50


def get_immediate_reward(old_state, state):
    # defining our own immediate rewards for the frozen lake model
    r = int(str(state.global_env['r'])[6:len(str(state.global_env['r'])) - 1])
    c = int(str(state.global_env['c'])[6:len(str(state.global_env['c'])) - 1])
    # reached goal
    if r == 0 and c == 5:
        return REW_GOAL
    # stepped into hole
    if (r == 3 and c == 1) or (r == 0 and c == 0) or (r == 4 and c == 2) or (r == 4 and c == 5) or (r == 2 and c == 5):
        return -REW_HOLE

    return 0


def get_value(state, variable_name):
    # Function is not relevant anymore, but might be useful for further development.
    # returns integer value of variable_name
    switch = {
        "r": int(str(state.global_env['r'])[6:len(str(state.global_env['r'])) - 1]),
        "c": int(str(state.global_env['c'])[6:len(str(state.global_env['c'])) - 1]),
    }
    return switch.get(variable_name, None)


def episode_finished(state):
    # episode is done if goal is reached
    r = get_value(state, "r")
    c = get_value(state, "c")
    # done when goal is reached
    goal = (r == 0 and c == 5)
    return goal
