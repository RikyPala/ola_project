from UCB import UCB
from Enviroment_UCB import Environment_UCB
import itertools
import numpy as np

ucb = UCB(5, 4)
env = Environment_UCB()
T = 1000
alpha_ratios = env.draw_alpha_ratios()


def find_optimal_arm_base(env):
    # print(''.join(map(str,configuration)))
    dict = {}
    for configuration in itertools.product([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]):
        configuration = np.asarray(configuration)
        reward = np.zeros(5)
        for i in range(len(configuration)):  # number of product
            common_term = env.conversion_rates[configuration[i]][i] * env.prices[i][configuration[i]] * env.product_sold[i][configuration[i]]
            reward[i] += env.expected_alpha_ratios[i]*common_term
        dict[''.join(map(str, configuration))] = np.sum(reward)

    return dict

def find_optimal_arm_until_lv1(env):
    # print(''.join(map(str,configuration)))
    dict = {}
    for configuration in itertools.product([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]):
        configuration = np.asarray(configuration)
        reward = np.zeros(5)
        for i in range(len(configuration)):  # number of product
            common_term = env.conversion_rates[configuration[i]][i] * env.prices[i][configuration[i]] * \
                          env.product_sold[i][configuration[i]]
            find_parents_lv1(env, i, configuration, reward, common_term)

            reward[i] += env.expected_alpha_ratios[i] * common_term
        dict[''.join(map(str, configuration))] = np.sum(reward)

    return dict

def find_optimal_arm_until_lv2(env):
    # print(''.join(map(str,configuration)))
    dict = {}
    for configuration in itertools.product([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]):
        configuration = np.asarray(configuration)
        reward = np.zeros(5)
        reward_tot = np.zeros(5)
        for i in range(len(configuration)):  # number of product
            common_term = env.conversion_rates[configuration[i]][i] * env.prices[i][configuration[i]] * \
                          env.product_sold[i][configuration[i]]
            parents_lv1 = find_parents_lv1(env, i, configuration, reward, common_term)
            find_parents_lv2(parents_lv1, env, i, configuration, reward, reward_tot)

            reward[i] += env.expected_alpha_ratios[i] * common_term
        dict[''.join(map(str, configuration))] = np.sum(reward)

    return dict

def find_optimal_arm_until_lv3(env):
    # print(''.join(map(str,configuration)))
    dict = {}
    for configuration in itertools.product([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]):

        configuration = np.asarray(configuration)
        print("CONFIGURATION")
        print(configuration)

        reward = np.zeros(5)
        reward_tot = np.zeros(5)

        for i in range(len(configuration)): # number of product
            common_term = env.conversion_rates[configuration[i]][i] * env.prices[i][configuration[i]] * env.product_sold[i][configuration[i]]
            reward[i] += env.expected_alpha_ratios[i]*common_term
            parents_lv1 = find_parents_lv1(env, i, configuration, reward, common_term)
            parents_lv2 = find_parents_lv2(parents_lv1, env, i, configuration, reward, reward_tot)
            find_parents_lv3(parents_lv2, env, i, configuration, reward, reward_tot)

        dict[''.join(map(str, configuration))] = np.sum(reward)

    return dict


def find_optimal_arm_until_lv4(env):
    # print(''.join(map(str,configuration)))
    dict = {}
    for configuration in itertools.product([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]):

        configuration = np.asarray(configuration)
        print("CONFIGURATION")
        print(configuration)

        reward = np.zeros(5)
        reward_tot = np.zeros(5)

        for i in range(len(configuration)):  # number of product
            common_term = env.conversion_rates[configuration[i]][i] * env.prices[i][configuration[i]] * \
                          env.product_sold[i][configuration[i]]
            reward[i] += env.expected_alpha_ratios[i] * common_term
            parents_lv1 = find_parents_lv1(env, i, configuration, reward, common_term)
            parents_lv2 = find_parents_lv2(parents_lv1, env, i, configuration, reward, reward_tot)
            parents_lv3 = find_parents_lv3(parents_lv2, env, i, configuration, reward, reward_tot)
            find_parents_lv4(parents_lv3, env, i, configuration, reward, reward_tot)

        dict[''.join(map(str, configuration))] = np.sum(reward)

    return dict


def find_optimal_arm_until_lv5(env):
    # print(''.join(map(str,configuration)))
    dict = {}
    for configuration in itertools.product([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]):

        configuration = np.asarray(configuration)
        print("CONFIGURATION")
        print(configuration)

        reward = np.zeros(5)
        reward_tot = np.zeros(5)

        for i in range(len(configuration)):  # number of product
            common_term = env.conversion_rates[configuration[i]][i] * env.prices[i][configuration[i]] * \
                          env.product_sold[i][configuration[i]]
            reward[i] += env.expected_alpha_ratios[i] * common_term
            parents_lv1 = find_parents_lv1(env, i, configuration, reward, common_term)
            parents_lv2 = find_parents_lv2(parents_lv1, env, i, configuration, reward, reward_tot)
            parents_lv3 = find_parents_lv3(parents_lv2, env, i, configuration, reward, reward_tot)
            parents_lv4 = find_parents_lv4(parents_lv3, env, i, configuration, reward, reward_tot)
            compute_parents_lv5(parents_lv4, env, i, configuration, reward, reward_tot)

        dict[''.join(map(str, configuration))] = np.sum(reward)

    return dict


def find_parents_lv1(env, i, configuration, reward, common_term):
    parents_lv1 = []
    for k in range(len(configuration)):
        if env.secondaries[k][0] == i:
            ca = env.conversion_rates[configuration[k]][k]*\
                         env.graph_probabilities[k][i]*common_term
            reward[i] += ca * env.expected_alpha_ratios[k]
            parents_lv1.append([k, i, ca])
        elif env.secondaries[k][1] == i:
            ca = env.conversion_rates[configuration[k]][k]* \
                         env.lambda_p*env.graph_probabilities[k][i]*common_term
            reward[i] += ca * env.expected_alpha_ratios[k]
            parents_lv1.append([k, i, ca])
    return parents_lv1


def find_parents_lv2(parents_lv1, env, i, configuration, reward, reward_tot):

    parents_lv2 = []
    for parent in parents_lv1:
        for k in range(len(configuration)):

            if env.secondaries[k][0] == parent[0] and k != parent[1]:

                ca = env.conversion_rates[configuration[k]][k]*env.graph_probabilities[k][parent[0]]*parent[2]
                reward_tot[i] += ca
                reward[i] += env.expected_alpha_ratios[k] * ca
                parents_lv2.append([k, parent[0], i, ca])

            elif env.secondaries[k][1] == parent[0] and k != parent[1]:

                ca = env.conversion_rates[configuration[k]][k] * env.graph_probabilities[k][parent[0]] * parent[2]
                reward[i] += env.expected_alpha_ratios[k]*ca
                parents_lv2.append([k, parent[0], i, ca])
                reward_tot[i] += ca

    return parents_lv2


def find_parents_lv3(parents_lv2, env, i, configuration, reward, reward_tot):
    parents_lv3 = []
    for parent in parents_lv2:
        for k in range(len(configuration)):

            if env.secondaries[k][0] == parent[0]  and k != parent[1] and k != parent[2]:

                ca = env.conversion_rates[configuration[k]][k] * env.graph_probabilities[k][parent[0]] * parent[3]
                reward_tot[i] += ca
                reward[i] += env.expected_alpha_ratios[k] * ca
                parents_lv3.append([k, parent[0], parent[1], i, ca])

            elif env.secondaries[k][1] == parent[0] and k != i and k != parent[1] and k != parent[2]:

                ca = env.conversion_rates[configuration[k]][k] * env.graph_probabilities[k][parent[0]] * parent[3]
                reward[i] += env.expected_alpha_ratios[k] * ca
                reward_tot[i] += ca
                parents_lv3.append([k, parent[0], parent[1], i, ca])
    return parents_lv3


def find_parents_lv4(parents_lv3, env, i, configuration, reward, reward_tot):
    parents_lv4 = []
    for parent in parents_lv3:
        for k in range(len(configuration)):

            if env.secondaries[k][0] == parent[0] and k != parent[1] and k != parent[2] and k!= parent[3]:

                ca = env.conversion_rates[configuration[k]][k] * env.graph_probabilities[k][parent[0]] * parent[4]
                reward_tot[i] += ca
                reward[i] += env.expected_alpha_ratios[k] * ca
                parents_lv4.append([k, parent[0], parent[1],parent[2], parent[3], ca])

            elif env.secondaries[k][1] == parent[0] and k != parent[1] and k != parent[2] and k!= parent[3]:

                ca = env.conversion_rates[configuration[k]][k] * env.graph_probabilities[k][parent[0]] * parent[4]
                reward[i] += env.expected_alpha_ratios[k] * ca
                reward_tot[i] += ca
                parents_lv4.append([k, parent[0], parent[1], parent[2], parent[3], ca])
    return parents_lv4


def compute_parents_lv5(parents_lv4, env, i, configuration, reward, reward_tot):

    for parent in parents_lv4:
        for k in range(len(configuration)):

            if env.secondaries[k][0] == parent[0] and k != parent[1] and k != parent[2] and k != parent[3] and k != parent[4]:

                ca = env.conversion_rates[configuration[k]][k] * env.graph_probabilities[k][parent[0]] * parent[5]
                reward_tot[i] += ca
                reward[i] += env.expected_alpha_ratios[k] * ca


            elif env.secondaries[k][1] == parent[0] and k != parent[1] and k != parent[2] and k != parent[3] and k != parent[4]:

                ca = env.conversion_rates[configuration[k]][k] * env.graph_probabilities[k][parent[0]] * parent[5]
                reward[i] += env.expected_alpha_ratios[k] * ca
                reward_tot[i] += ca

while ucb.t < T:

    pulled_arms = ucb.pull_arm(env.prices, env.product_sold)
    print("PULLED")
    print(pulled_arms)
    conversion = env.round(pulled_arms)
    print("CONVERSIONNNNNNNNNNNNNNNNNNNNNNN")
    print(conversion)
    ucb.update(pulled_arms, conversion, env.graph_probabilities, env.secondaries, env.lambda_p)

dict0 = find_optimal_arm_base(env)
dict1 = find_optimal_arm_until_lv1(env)
dict2 = find_optimal_arm_until_lv2(env)
dict3 = find_optimal_arm_until_lv3(env)
dict4 = find_optimal_arm_until_lv4(env)
dict5 = find_optimal_arm_until_lv5(env)

Keymax0 = max(zip(dict0.values(), dict0.keys()))[1]
print(Keymax0)
print(dict0)

Keymax1 = max(zip(dict1.values(), dict1.keys()))[1]
print(Keymax1)
print(dict1)

Keymax2 = max(zip(dict2.values(), dict2.keys()))[1]
print(Keymax2)
print(dict2)

Keymax3 = max(zip(dict3.values(), dict3.keys()))[1]
print(Keymax3)
print(dict3)

Keymax4 = max(zip(dict4.values(), dict4.keys()))[1]
print(Keymax4)
print(dict4)

Keymax5 = max(zip(dict5.values(), dict5.keys()))[1]
print(Keymax5)
print(dict5)





