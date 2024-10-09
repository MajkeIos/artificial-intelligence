import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
import sys

MAX_CARS = 20
DISCOUNT_RATE = 0.9
RENTAL_REWARD = 10
MOVING_REWARD = -2


class Poisson_:
    def __init__(self, lambda_):
        self.lambda_ = lambda_
        epsilon = 0.01
        self.left_ = 0
        state = 1
        self.vals = {}
        summer = 0
        while True:
            if state == 1:
                temp = poisson.pmf(self.left_, self.lambda_)
                if temp <= epsilon:
                    self.left_ += 1
                else:
                    self.vals[self.left_] = temp
                    summer += temp
                    self.right_ = self.left_ + 1
                    state = 2
            elif state == 2:
                temp = poisson.pmf(self.right_, self.lambda_)
                if temp > epsilon:
                    self.vals[self.right_] = temp
                    summer += temp
                    self.right_ += 1
                else:
                    break
        added_val = (1 - summer) / (self.right_ - self.left_)
        for key in self.vals:
            self.vals[key] += added_val

    def f(self, n):
        try:
            Ret_value = self.vals[n]
            return Ret_value
        except KeyError:
            Ret_value = 0
            return Ret_value


class Location:
    def __init__(self, requests, returns):
        self.poissonRequest = Poisson_(requests)
        self.poissonReturn = Poisson_(returns)


def apply_action(state, action):
    return [max(min(state[0] - action, MAX_CARS), 0), max(min(state[1] + action, MAX_CARS), 0)]


def expected_reward(state, action):
    global value
    reward = 0
    new_state = apply_action(state, action)
    reward = reward + MOVING_REWARD * abs(action)

    for aRequests in range(A.poissonRequest.left_, A.poissonRequest.right_):
        for bRequests in range(B.poissonRequest.left_, B.poissonRequest.right_):
            for aReturns in range(A.poissonReturn.left_, A.poissonReturn.right_):
                for bReturns in range(B.poissonReturn.left_, B.poissonReturn.right_):
                    probability = A.poissonRequest.vals[aRequests] * B.poissonRequest.vals[bRequests] * \
                                  A.poissonReturn.vals[aReturns] * B.poissonReturn.vals[bReturns]
                    valid_requests_A = min(new_state[0], aRequests)
                    valid_requests_B = min(new_state[1], bRequests)

                    cur_reward = (valid_requests_A + valid_requests_B) * RENTAL_REWARD
                    new_s = [0, 0]
                    new_s[0] = max(min(new_state[0] - valid_requests_A + aReturns, MAX_CARS), 0)
                    new_s[1] = max(min(new_state[1] - valid_requests_B + bReturns, MAX_CARS), 0)

                    reward += probability * (cur_reward + DISCOUNT_RATE * value[new_s[0]][new_s[1]])
    return reward


def policy_evaluation(epsilon):
    global value
    while True:
        delta = 0
        for i in range(value.shape[0]):
            for j in range(value.shape[1]):
                old_val = value[i][j]
                value[i][j] = expected_reward([i, j], policy[i][j])
                delta = max(delta, abs(value[i][j] - old_val))
        print(delta)
        sys.stdout.flush()
        if delta < epsilon:
            break


def policy_improvement():
    global policy
    policy_stable = True
    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
            old_action = policy[i][j]
            max_act_val = None
            max_act = None
            c12 = min(i, 5)
            c21 = -min(j, 5)
            for act in range(c21, c12 + 1):
                omega = expected_reward([i, j], act)
                if max_act_val is None:
                    max_act_val = omega
                    max_act = act
                elif max_act_val < omega:
                    max_act_val = omega
                    max_act = act
            policy[i][j] = max_act
            if old_action != policy[i][j]:
                policy_stable = False
    return policy_stable


def save_value():
    ax = sns.heatmap(value, linewidth=1)
    ax.invert_yaxis()
    plt.savefig('value.svg')
    plt.close()


def save_policy():
    ax = sns.heatmap(policy, linewidth=1, annot=True)
    ax.invert_yaxis()
    plt.savefig('policy.svg')
    plt.close()


A = Location(3, 3)
B = Location(4, 2)
value = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
policy = value.copy().astype(int)
pol_eval_epsilon = 50

while True:
    policy_evaluation(pol_eval_epsilon)
    stable = policy_improvement()
    if stable:
        save_value()
        save_policy()
        break
    print('')
    pol_eval_epsilon /= 10
