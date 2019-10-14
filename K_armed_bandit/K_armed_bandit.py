"""
a implementation of K armed bandit, the algorithm is in page 375 of book <Machine Learning>
"""
import random

R = [1, 2, 3, 4, 5]
e = 0.5
r = 0
arms_num = 5
Q = [0 for i in range(arms_num)]
count = [0 for i in range(arms_num)]
T = 100000
for i in range(T):
    p = random.random()
    if p <0.001:
        k = random.randint(0, 4)
    else:
        temp = max(Q)
        max_index = []
        for j in range(len(Q)):
            if Q[j] == temp:
                max_index.append(j)
        index = random.randint(0, len(max_index) - 1)
        k = max_index[index]
    v = R[k]
    r += v
    Q[k] = (Q[k] * count[k] + v) / (count[k] + 1)
    count[k] += 1
print(r)
print(count)
