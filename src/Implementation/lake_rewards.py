def get_immediate_reward(old_state, state):
    r = int(str(state.global_env['r'])[6:len(str(state.global_env['r'])) - 1])
    c = int(str(state.global_env['c'])[6:len(str(state.global_env['c'])) - 1])

    # reached goal
    if r == 0 and c == 5:
        return 100
    # stepped into hole
    if (r ==3 and c ==1) or (r == 0 and c == 0) or (r ==4 and c ==2) or (r == 4 and c == 5) or (r ==2  and c == 5):
        return -50

    return 0