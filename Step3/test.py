from Environment import Environment
import numpy as np


Env = Environment()
print(Env.alpha_ratios_parameters[:,:,0])
ratios = Env.draw_alpha_ratios()
print(ratios)
ratios = np.reshape(ratios, -1)
print(ratios)
print(Env.draw_starting_page(ratios))
