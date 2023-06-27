#!/usr/bin/env python
import copy

def convert_keys(prefix_change_list, state):
    new_state = copy.deepcopy(state)
    for s in prefix_change_list:
        ss = s.split('->', 1)
        for k in state:
            if k.startswith(ss[0]):
                new_k = k.replace(ss[0], ss[1])
                # logging.info(f'Convert {k} to {new_k}.')
                new_state.pop(k)
                new_state[new_k] = state[k]
    return new_state

def filter_state(state1, state2, prefix_change_list=[]):

    dismatching_keys = []
    missing_keys = []
    unexpected_keys = []
    match_state = {}

    state2 = convert_keys(prefix_change_list, state2)

    for k in state1:
        if k not in state2:
            missing_keys.append(k)
            continue
        if state1[k].shape != state2[k].shape:
            dismatching_keys.append((k, state1[k].shape, state2[k].shape))
        else:
            match_state[k] = state2[k]
    for k in state2:
        if k not in state1:
            unexpected_keys.append(k)
    return match_state, dismatching_keys, missing_keys, unexpected_keys
