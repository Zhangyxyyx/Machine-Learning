"""
a implementation of Sarsa and Q-learning in TD reinforcement learning, the algorithm is in page 388 of book <Machine Learning>
"""
import random

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
X_max_A = [1, 0, 0, 0]


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

def sarsa(E, A, x0, reward_discount, lr,T):
    probability = 0.5
    Q = [[0 for i in range(len(A))] for j in range(len(X))]
    x=x0
    a=X_max_A[x]
    for t in range(T):
        x_,reward=E(x,a)
        a_=policy(x_,probability)
        Q[x][a]=Q[x][a]+lr*(reward+reward_discount*Q[x_][a_]-Q[x][a])
        for x in range(len(X)):
            action = Q[x].index(max(Q[x]))
            X_max_A[x] = action
        x=x_
        a=a_

def q_learning(E, A, x0, reward_discount, lr,T):
    probability = 0.5
    Q = [[0 for i in range(len(A))] for j in range(len(X))]
    x=x0
    for t in range(T):
        a=policy(x,probability)
        x_,reward=E(x,a)
        a_=X_max_A[x_]
        Q[x][a]=Q[x][a]+lr*(reward+reward_discount*Q[x_][a_]-Q[x][a])
        for x in range(len(X)):
            action = Q[x].index(max(Q[x]))
            X_max_A[x] = action
        x=x_



if __name__=='__main__':
    acc=0
    for i in range(100):
        X_max_A = [1, 0, 0, 0]
        q_learning(environment,A,0,0.1,0.1,100000)
        if X_max_A==[1,1,0,0] or X_max_A==[1,1,0,1]:
            acc+=1
        print(X_max_A)
    print(acc)