from cgitb import enable
from winreg import EnumValue
import numpy as np
from matplotlib import pyplot as plt
from Environment import Environment
from Learner import Learner

env = Environment()
learner = Learner(env)

learner.greedy_optimization()
