import pprint
import numpy as np

# Markov Decision Process - Value Iteration

# the agent is in a n*m labyrinth
# he cannot exit the labyrinth or move in diagonal
n = m = 5

# set the goal cell
goal = (4, 3)

# Definition of the MDP

# Set of states
states = [(i, j) for i in range(5) for j in range(5)]

# Set of actions
actions = ["left", "right", "up", "down"]


# gives the next state of the agent
def transition(state, action):
    i, j = state
    if action == "left":
        return i, max(0, j - 1)
    elif action == "right":
        return i, min(n - 1, j + 1)
    elif action == "up":
        return max(0, i - 1), j
    elif action == "down":
        return min(m - 1, i + 1), j


# give the reward to the agent when he changes his state
def reward(state):
    if state == goal:
        return 10
    else:
        return -1


# Value Iteration

# Initial policy (random action with equal probability)
# for each state the probability to move to the other possible
# states are equal
policy = {state: {a: 1 / len(actions) for a in actions} for state in states}


def value_iteration():
    # Initialize the values of all states to zero
    values = {state: 0 for state in states}
    # Set the discount factor
    gamma = 0.9
    # Repeat until the values converge
    while True:
        delta = 0
        for state in states:
            old_value = values[state]
            new_value = max(reward(state) + gamma * values[transition(state, action)] for action in actions)
            values[state] = new_value
            delta = max(delta, abs(new_value - old_value))
        if delta < 1e-9:
            break

    # Derive the optimal policy from the values
    better_policy = {state: max(actions, key=lambda a: reward(state) + gamma * values[transition(state, a)]) for state
                     in states}
    return better_policy, values

pprint.pprint(value_iteration())
