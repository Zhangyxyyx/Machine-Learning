"""
a implementation of model-free-learning, the algorithm is in page 382 of book <Machine Learning>
"""
import random
import functools
import operator

# MDP E=<X,A,P,R> means <states, actions, states transfer probabilities, rewards>,
# the model is about planting flower on page 372

A = [0, 1]  # using 0 and 1 represent 'not water' and 'water' respectively


def environment(state, action):
    if state == 0:
        if action == 1:
            p = random.random()
            if p < 0.6:
                state = 0
                reward = 10
            else:
                state = 2
                reward = -10
        elif action == 0:
            p = random.random()
            if p < 0.4:
                state = 0
                reward = 10
            else:
                state = 1
                reward = -10
    elif state == 1:
        if action == 1:
            p = random.random()
            if p < 0.5:
                state = 1
                reward = -10
            else:
                state = 0
                reward = 10
        elif action == 0:
            p = random.random()
            if p < 0.4:
                state = 1
                reward = -10
            else:
                state = 3
                reward = -1000
    elif state == 2:
        if action == 1:
            p = random.random()
            if p < 0.4:
                state = 2
                reward = -10
            else:
                state = 3
                reward = -1000
        elif action == 0:
            p = random.random()
            if p < 0.6:
                state = 0
                reward = 10
            else:
                state = 2
                reward = -10
    elif state == 3:
        state = 3
        reward = -1000
    return state, reward


X = [0, 1, 2, 3]
X_max_A = [0, 0, 0, 0]


def policy(state, probability):
    p = random.random()
    if p < probability:
        t = random.random()
        if t < 0.5:
            return 0
        else:
            return 1
    else:
        return X_max_A[state]


# P is the state transfer probability matrix P[state2][action][state2]=probability
# P = [
#     [[0.4, 0.6, 0, 0], [0.6, 0, 0.4, 0]],
#     [[0, 0.4, 0, 0.6], [0.5, 0.5, 0, 0]],
#     [[0.6, 0, 0.4, 0], [0, 0, 0.6, 0.4]],
#     [[0, 0, 0, 1], [0, 0, 0, 1]]
# ]
#
# # R is the state transfer reward matrix R[state2][action][state2]=reward
# R = [
#     [[10, -10, 0, 0], [10, 0, -10, 0]],
#     [[0, -10, 0, -100], [10, -10, 0, 0]],
#     [[10, 0, -10, 0], [0, 0, -10, -100]],
#     [[0, 0, 0, -100], [0, 0, 0, -100]]
# ]

def on_policy(E, A, state0, T, S):
    probability = 0.2
    Q = [[0 for i in range(len(A))] for j in range(len(X))]
    max_Q = [-100000000 for j in range(len(X))]
    count = [[0 for i in range(len(A))] for j in range(len(X))]
    for s in range(S):
        x = []
        a = []
        r = []
        x.append(state0)
        state = state0
        action = policy(state0, probability)
        a.append(action)
        for t in range(T):
            state, reward = E(state, action)
            r.append(reward)
            x.append(state)
            action = policy(state, probability)
            a.append(action)
        x.pop(-1)
        a.pop(-1)
        for t in range(T):
            temp = r[t + 1:]
            R = sum(temp) / (T - t)
            Q[x[t]][a[t]] = ((Q[x[t]][a[t]] * count[x[t]][a[t]]) + R) / (count[x[t]][a[t]] + 1)
            count[x[t]][a[t]] += 1
        for x in range(len(X)):
            action=Q[x].index(max(Q[x]))
            X_max_A[x]=action

        print(Q)


def off_policy(E, A, state0, T, S):
    probability = 0.2
    Q = [[0 for i in range(len(A))] for j in range(len(X))]
    max_Q = [-100000000 for j in range(len(X))]
    count = [[0 for i in range(len(A))] for j in range(len(X))]
    for s in range(S):
        x = []
        a = []
        r = []
        p = []
        x.append(state0)
        state = state0
        action = policy(state0, probability)
        a.append(action)
        for t in range(T):
            if action == X_max_A[state]:
                pi = 1 - probability + (probability / 2)
            else:
                pi = probability / 2
            p.append(pi)
            state, reward = E(state, action)
            r.append(reward)
            x.append(state)
            action = policy(state, probability)
            a.append(action)
        x.pop(-1)
        a.pop(-1)
        for t in range(T):
            ratio = 1
            for j in range(t + 1, T - 1):
                if a[j] == X_max_A[x[j]]:
                    ratio*=1/p[j]
                else:
                    ratio=0
                    break
            R = (sum(r[t + 1:]) / (T - t))*ratio
            Q[x[t]][a[t]] = ((Q[x[t]][a[t]] * count[x[t]][a[t]]) + R) / (count[x[t]][a[t]] + 1)
            count[x[t]][a[t]] += 1
        for x in range(len(X)):
            action=Q[x].index(max(Q[x]))
            X_max_A[x]=action

        print(max_Q)


if __name__ == '__main__':
    print(X_max_A)
    on_policy(environment, A, 0, 10000, 10)
    print(X_max_A)
