from TS import TS
import numpy as np
import matplotlib.pyplot as plt

learner = TS(4, prior_mean = 200)
mu = [100.0, 110.0, 120.0, 130.0]
v = [100.0, 100.0, 100.0, 100.0]

for i in range(10000):
    best = learner.pull()
    reward = np.random.normal(mu[best], np.sqrt(v[best]))
    learner.update(reward)
    print(str(best) + " : " + str(reward))
