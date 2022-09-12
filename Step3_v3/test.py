import numpy as np

from Environment import Environment
from OptimizerTS import OptimizerTS


env = Environment()
learner = OptimizerTS(env)

print(learner.alpha_ratios)
alpha = np.sum(env.draw_alpha_ratios() * np.expand_dims(env.user_probabilities, axis=1), axis=0)
print(alpha)
