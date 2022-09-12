from TS import TS
import numpy as np
import matplotlib.pyplot as plt

learner = TS((4, ), prior_mean=200 * 5)
mu = [100.0, 110.0, 120.0, 130.0]
v = [100.0, 100.0, 100.0, 100.0]

for i in range(10000):
    best = learner.pull()
    reward = 0
    for x in best:
        reward += np.random.normal(mu[x], np.sqrt(v[x]))
    learner.update(reward)
    print(str(best) + " : " + str(reward))
