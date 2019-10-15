"""
a implementation of model-based-learning, the algorithm is in page 379 and 381 of book <Machine Learning>
"""
import random

# MDP E=<X,A,P,R> means <states, actions, states transfer probabilities, rewards>, the model is about planting flower on page 372

A = {}
A[0] = 'not water'
A[1] = 'water'  # using 0 and 1 represent 'not water' and 'water' respectively

policy = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]

# def P_and_R(state, action):
#     if state == 0:
#         if action == 1:
#             p = random.random()
#             if p < 0.6:
#                 state = 0
#                 reward = 1
#             else:
#                 state = 2
#                 reward = -1
#         elif action == 0:
#             p = random.random()
#             if p < 0.4:
#                 state = 0
#                 reward = 1
#             else:
#                 state = 1
#                 reward = -1
#     elif state == 1:
#         if action == 1:
#             p = random.random()
#             if p < 0.5:
#                 state = 1
#                 reward = -1
#             else:
#                 state = 0
#                 reward = 1
#         elif action == 0:
#             p = random.random()
#             if p < 0.4:
#                 state = 1
#                 reward = -1
#             else:
#                 state = 3
#                 reward = -100
#     elif state == 2:
#         if action == 1:
#             p = random.random()
#             if p < 0.6:
#                 state = 2
#                 reward = -1
#             else:
#                 state = 3
#                 reward = -100
#         elif action == 0:
#             p = random.random()
#             if p < 0.6:
#                 state = 0
#                 reward = 1
#             else:
#                 state = 2
#                 reward = -1
#     elif state == 3:
#         state == 3
#         reward = -100
#     return state, reward


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


def policy_evaluation_in_Tsteps(T,policy):
    t = 1
    # state value function,there are 4 states, using 0,1,2,3 represent 'health','out of
    # water','too much water',and 'dead' respectively
    V = [0 for i in range(4)]
    state=[]
    action=[]
    while True:
        for x in range(len(V)):
            v = 0
            max_Q = float("-inf")
            for a in range(len(A.keys())):
                Q = 0
                for s2 in range(len(V)):
                    Q += P[x][a][s2] * ((1 / t) * R[x][a][s2] + ((t - 1) / t) * V[s2])
                if Q>max_Q:
                    max_Q=Q
                    state.append(x)
                    action.append(a)
                v += policy[x][a] * Q
            V[x] = v
        t += 1
        if t == T + 1:
            break
    return V,state,action


def value_iteration(T, threshold,policy):
    t = 1
    # state value function,there are 4 states, using 0,1,2,3 represent 'health','out of
    # water','too much water',and 'dead' respectively
    V = [0 for i in range(4)]
    max_diff = 0
    while True:
        for x in range(len(V)):
            v = 0
            for a in range(len(A.keys())):
                max_Q =float("-inf")
                for s2 in range(len(V)):
                    Q = P[x][a][s2] * ((1 / t) * R[x][a][s2] + ((t - 1) / t) * V[s2])
                    if Q > max_Q:
                        max_Q = Q
                        policy[x][a]=1
                        policy[x][1-a]=0
                        print("state: {} action: {} max_Q: {:.6f}".format(x,a,max_Q))
                v = max_Q
            max_diff = max(max_diff, v-V[x])
            V[x] = v
        print(max_diff)
        if max_diff<threshold:
            break
    return policy

def value_iteration2(T, threshold,policy):
    t = 1
    # state value function,there are 4 states, using 0,1,2,3 represent 'health','out of
    # water','too much water',and 'dead' respectively
    V = [0 for i in range(4)]
    max_diff = 0
    while True:
        value,state,action=policy_evaluation_in_Tsteps(T,policy)
        print(value)
        policy_=policy
        for index in zip(state,action):
            policy[index[0]][index[1]]=1
            policy[index[0]][1-index[1]]=0
        if policy==policy_:
            break
    return policy


if __name__ == '__main__':
    value, state, action = policy_evaluation_in_Tsteps(1, policy)
    print(value)
    p=value_iteration2(100,1,policy)
    print(p)
    value, state, action = policy_evaluation_in_Tsteps(1, p)
    print(value)



