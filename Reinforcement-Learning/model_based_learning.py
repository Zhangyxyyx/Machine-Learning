"""
a implementation of model-based-learning, the algorithm is in page 379 and 381 of book <Machine Learning>
"""
import random

# MDP E=<X,A,P,R> means <states, actions, states transfer probabilities, rewards>, the model is about planting flower on page 372

A = {}
A[0] = 'not water'
A[1] = 'water'  # using 0 and 1 represent 'not water' and 'water' respectively

policy = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]

X = [0, 1, 2, 3]

# P is the state transfer probability matrix P[state2][action][state2]=probability
P = [
    [[0.4, 0.6, 0, 0], [0.6, 0, 0.4, 0]],
    [[0, 0.4, 0, 0.6], [0.5, 0.5, 0, 0]],
    [[0.6, 0, 0.4, 0], [0, 0, 0.6, 0.4]],
    [[0, 0, 0, 1], [0, 0, 0, 1]]
]

# R is the state transfer reward matrix R[state2][action][state2]=reward
R = [
    [[10, -10, 0, 0], [10, 0, -10, 0]],
    [[0, -10, 0, -100], [10, -10, 0, 0]],
    [[10, 0, -10, 0], [0, 0, -10, -100]],
    [[0, 0, 0, -100], [0, 0, 0, -100]]
]


def policy_evaluation_in_Tsteps(T, policy):
    t = 1
    # state value function,there are 4 states, using 0,1,2,3 represent 'health','out of
    # water','too much water',and 'dead' respectively
    V = [0 for i in range(4)]
    while True:
        for x in range(len(V)):
            v = 0
            for a in range(len(A.keys())):
                Q = 0
                for s2 in range(len(V)):
                    Q += P[x][a][s2] * ((1 / t) * R[x][a][s2] + ((t - 1) / t) * V[s2])
                v += policy[x][a] * Q
            V[x] = v
        t += 1
        if t == T + 1:
            break
    return V


def value_iteration(T, threshold, policy):
    t = 1
    # state value function,there are 4 states, using 0,1,2,3 represent 'health','out of
    # water','too much water',and 'dead' respectively
    V = [0 for i in range(4)]
    V_=V
    diff=V
    while True:
        for x in range(len(V)):
            v=0
            for a in range(len(A.keys())):
                max_v = float("-inf")
                for s2 in range(len(V)):
                    v += P[x][a][s2] * ((1 / t) * R[x][a][s2] + ((t - 1) / t) * V[s2])
                if v > max_v:
                    max_v = v
            V_[x] = max_v
            diff[x]=abs(V_[x]-V[x])
        if max(diff)<threshold:
            break
        else:
            V=V_
            t+=1

    Q = [[0 for i in range(len(A.keys()))] for j in range(len(V))]
    for x in range(len(V)):
        for a in range(len(A.keys())):
            for s in range(len(V)):
                Q[x][a] += P[x][a][s] * ((1 / T) * R[x][a][s] + (T - 1) / T * V[x])
        action = Q[x].index(max(Q[x]))
        policy[x][action] = 1
        policy[x][1 - action] = 0
    return policy


def policy_iteration(T, policy):
    t = 1
    # state value function,there are 4 states, using 0,1,2,3 represent 'health','out of
    # water','too much water',and 'dead' respectively
    max_diff = 0
    while True:
        V = policy_evaluation_in_Tsteps(T, policy)
        Q = [[0 for i in range(len(A.keys()))] for j in range(len(V))]
        policy_ = policy
        for x in range(len(V)):
            for a in range(len(A.keys())):
                for s in range(len(V)):
                    Q[x][a] += P[x][a][s] * ((1 / T) * R[x][a][s] + (T - 1) / T * V[x])
            action = Q[x].index(max(Q[x]))
            policy_[x][action] = 1
            policy_[x][1 - action] = 0
        if policy == policy_:
            break
        else:
            policy = policy_
    return policy


if __name__ == '__main__':
    print(policy)
    p = value_iteration(100,1, policy)
    print(policy)
