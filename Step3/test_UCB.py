from UCB import UCB
from Enviroment_UCB import Environment_UCB





ucb = UCB(5,4)
env = Environment_UCB()
while ucb.t < 4:
    pulled_arms = ucb.pull_arm()
    print(pulled_arms)
    rewards, conversion, number = env.round(pulled_arms)
    ucb.update(pulled_arms, conversion)

    print(conversion)
    #print(number)



