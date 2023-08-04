"""
MDP algorithms from Russel and Norvig's AI textbook. Without Grid_MDP class
"""
import random
from collections import defaultdict

import numpy as np

# 16.4 Partially Observed MDPs

class POMDP:
    """A Partially Observable Markov Decision Process, defined by
    a transition model P(s'|s,a), actions A(s), a reward function R(s),
    and a sensor model P(e|s). We also keep track of a gamma value,
    for use by algorithms. The transition and the sensor models
    are defined as matrices. We also keep track of the possible states
    and actions for each state. [Page 659]."""

    def __init__(self, actions, transitions=None, evidences=None, rewards=None, states=None, gamma=0.95):
        """Initialize variables of the pomdp"""

        if not (0 < gamma <= 1):
            raise ValueError('A POMDP must have 0 < gamma <= 1')

        self.states = states
        self.actions = actions

        # transition model cannot be undefined
        self.t_prob = transitions or {}
        if not self.t_prob:
            print('Warning: Transition model is undefined')

        # sensor model cannot be undefined
        self.e_prob = evidences or {}
        if not self.e_prob:
            print('Warning: Sensor model is undefined')

        self.gamma = gamma
        self.rewards = rewards

    def remove_dominated_plans(self, input_values):
        """
        Remove dominated plans.
        This method finds all the lines contributing to the
        upper surface and removes those which don't.
        """

        values = [val for action in input_values for val in input_values[action]]
        values.sort(key=lambda x: x[0], reverse=True)

        best = [values[0]]
        y1_max = max(val[1] for val in values)
        tgt = values[0]
        prev_b = 0
        prev_ix = 0
        while tgt[1] != y1_max:
            min_b = 1
            min_ix = 0
            for i in range(prev_ix + 1, len(values)):
                if values[i][0] - tgt[0] + tgt[1] - values[i][1] != 0:
                    trans_b = (values[i][0] - tgt[0]) / (values[i][0] - tgt[0] + tgt[1] - values[i][1])
                    if 0 <= trans_b <= 1 and trans_b > prev_b and trans_b < min_b:
                        min_b = trans_b
                        min_ix = i
            prev_b = min_b
            prev_ix = min_ix
            tgt = values[min_ix]
            best.append(tgt)

        return self.generate_mapping(best, input_values)

    def remove_dominated_plans_fast(self, input_values):
        """
        Remove dominated plans using approximations.
        Resamples the upper boundary at intervals of 100 and
        finds the maximum values at these points.
        """

        values = [val for action in input_values for val in input_values[action]]
        values.sort(key=lambda x: x[0], reverse=True)

        best = []
        sr = 100
        for i in range(sr + 1):
            x = i / float(sr)
            maximum = (values[0][1] - values[0][0]) * x + values[0][0]
            tgt = values[0]
            for value in values:
                val = (value[1] - value[0]) * x + value[0]
                if val > maximum:
                    maximum = val
                    tgt = value

            if all(any(tgt != v) for v in best):
                best.append(np.array(tgt))

        return self.generate_mapping(best, input_values)

    def generate_mapping(self, best, input_values):
        """Generate mappings after removing dominated plans"""

        mapping = defaultdict(list)
        for value in best:
            for action in input_values:
                if any(all(value == v) for v in input_values[action]):
                    mapping[action].append(value)

        return mapping

    def max_difference(self, U1, U2):
        """Find maximum difference between two utility mappings"""

        for k, v in U1.items():
            sum1 = 0
            for element in U1[k]:
                sum1 += sum(element)
            sum2 = 0
            for element in U2[k]:
                sum2 += sum(element)
        return abs(sum1 - sum2)


class Matrix:
    """Matrix operations class"""

    @staticmethod
    def add(A, B):
        """Add two matrices A and B"""

        res = []
        for i in range(len(A)):
            row = []
            for j in range(len(A[0])):
                row.append(A[i][j] + B[i][j])
            res.append(row)
        return res

    @staticmethod
    def scalar_multiply(a, B):
        """Multiply scalar a to matrix B"""

        for i in range(len(B)):
            for j in range(len(B[0])):
                B[i][j] = a * B[i][j]
        return B

    @staticmethod
    def multiply(A, B):
        """Multiply two matrices A and B element-wise"""

        matrix = []
        for i in range(len(B)):
            row = []
            for j in range(len(B[0])):
                row.append(B[i][j] * A[j][i])
            matrix.append(row)

        return matrix

    @staticmethod
    def matmul(A, B):
        """Inner-product of two matrices"""

        return [[sum(ele_a * ele_b for ele_a, ele_b in zip(row_a, col_b)) for col_b in list(zip(*B))] for row_a in A]

    @staticmethod
    def transpose(A):
        """Transpose a matrix"""

        return [list(i) for i in zip(*A)]


def pomdp_value_iteration(pomdp, epsilon=0.1):
    """Solving a POMDP by value iteration."""

    U = {'': [[0] * len(pomdp.states)]}
    count = 0
    while True:
        count += 1
        prev_U = U
        values = [val for action in U for val in U[action]]
        value_matxs = []
        for i in values:
            for j in values:
                value_matxs.append([i, j])

        U1 = defaultdict(list)
        for action in pomdp.actions:
            for u in value_matxs:
                u1 = Matrix.matmul(Matrix.matmul(pomdp.t_prob[int(action)],
                                                 Matrix.multiply(pomdp.e_prob[int(action)], Matrix.transpose(u))),
                                   [[1], [1]])
                u1 = Matrix.add(Matrix.scalar_multiply(pomdp.gamma, Matrix.transpose(u1)), [pomdp.rewards[int(action)]])
                U1[action].append(u1[0])

        U = pomdp.remove_dominated_plans_fast(U1)
        # replace with U = pomdp.remove_dominated_plans(U1) for accurate calculations

        if count > 10:
            if pomdp.max_difference(U, prev_U) < epsilon * (1 - pomdp.gamma) / pomdp.gamma:
                return U


__doc__ += """
>>> pi = best_policy(sequential_decision_environment, value_iteration(sequential_decision_environment, .01))

>>> sequential_decision_environment.to_arrows(pi)
[['>', '>', '>', '.'], ['^', None, '^', '.'], ['^', '>', '^', '<']]

>>> from utils import print_table

>>> print_table(sequential_decision_environment.to_arrows(pi))
>   >      >   .
^   None   ^   .
^   >      ^   <

>>> print_table(sequential_decision_environment.to_arrows(policy_iteration(sequential_decision_environment)))
>   >      >   .
^   None   ^   .
^   >      ^   <
"""  # noqa

"""
s = { 'a' : {	'plan1' : [(0.2, 'a'), (0.3, 'b'), (0.3, 'c'), (0.2, 'd')],
                'plan2' : [(0.4, 'a'), (0.15, 'b'), (0.45, 'c')],
                'plan3' : [(0.2, 'a'), (0.5, 'b'), (0.3, 'c')],
                 },
      'b' : {	'plan1' : [(0.2, 'a'), (0.6, 'b'), (0.2, 'c'), (0.1, 'd')],
                'plan2' : [(0.6, 'a'), (0.2, 'b'), (0.1, 'c'), (0.1, 'd')],
                'plan3' : [(0.3, 'a'), (0.3, 'b'), (0.4, 'c')],
                },
        'c' : {	'plan1' : [(0.3, 'a'), (0.5, 'b'), (0.1, 'c'), (0.1, 'd')],
                'plan2' : [(0.5, 'a'), (0.3, 'b'), (0.1, 'c'), (0.1, 'd')],
                'plan3' : [(0.1, 'a'), (0.3, 'b'), (0.1, 'c'), (0.5, 'd')],
                },
    }
"""
