import numpy as np

from Environment import Environment
from Solver import Solver


env = Environment()
solver = Solver(env)

solver.optimize()
