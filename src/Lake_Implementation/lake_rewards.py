

def get_immediate_reward(old_state, state):
    # TODO
    r = int(str(state.global_env['r'])[6:len(str(state.global_env['r'])) - 1])
    
    c = int(str(state.global_env['c'])[6:len(str(state.global_env['c'])) - 1])

    if r == 0 and c == 5:
        return 100

    if (r ==3 and c ==1) or (r == 0 and c == 0) or (r == 4 and c == 5) or (r ==2  and c == 5):
        return -50

    return 0