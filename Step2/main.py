from Environment import Environment
from Learner import Learner

env = Environment()
learner = Learner(env)

learner.greedy_optimization()
