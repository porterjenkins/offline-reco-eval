import pandas as pd
import numpy as np
from datetime import datetime

class RandomPolicy(object):

    def __init__(self, products: list, range: tuple):
        self.product = products
        self.q_min, self.q_max = range
        self.n_min, self.n_max = 1, 10


    def __call__(self, state):
        n_prod = np.random.randint(self.n_min, self.n_max)
        prods = np.random.choice(self.product, size=n_prod)
        a = {}
        for p in prods:
            q = np.random.randint(self.q_min, self.q_max)
            a[p] = q
        return a



class DisplayState(object):

    def __init__(self, prod_quantity: dict, max_slots: int, timestamp: datetime):
        self.max_slots = max_slots
        self.ts = timestamp
        self.prods = set(prod_quantity.keys())
        self.quantities = prod_quantity



class OfflineDisplayPolicyEvaluator(object):

    """
    Li et al. 2011
    'Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms
    """

    def __init__(self, df: pd.DataFrame):
        self.events = self.get_events(df)
        self.curr = 0
        self.is_valid_cnt = 0

    @staticmethod
    def get_events(df: pd.DataFrame):
        events = []
        groups = df[['last_scanned_datetime', 'name', 'previous_post_scan_num_facings', 'payoff', 'max_slots']].groupby('last_scanned_datetime')
        prev_state = None
        i = 0
        for k, v in groups:
            state = prev_state
            action = dict(zip(v['name'], v['previous_post_scan_num_facings']))
            payoff = dict(zip(v['name'], v['payoff']))
            if i > 0:
                # skip initial action (no prev state available)
                events.append(
                    {'state': state, 'payoff': payoff, "action": action, "timestamp": k}
                )

            prev_state = DisplayState(
                prod_quantity=action,
                timestamp=v['last_scanned_datetime'].max(),
                max_slots=v['max_slots'].max()
            )
            i += 1

        return events

    def step(self, a):
        """
            input: action
            output: reward, new_state
        """
        event = self.__getitem__(self.curr)
        true_action = event['action']
        payoff = event['payoff']

        reward = self._get_reward(
            true_action=true_action,
            action=a,
            true_payoffs=payoff
        )

        new_state = self._increment()
        return reward, new_state, payoff

    def _get_reward(self, true_action: dict, action: dict, true_payoffs: dict):

        # TODO: consider adding penalty for max_slots

        r = 0

        for a in action.keys():
            if a in true_action:
                # check if quantities are equal
                if true_action[a] == action[a]:
                    # add payoff
                    r += true_payoffs[a]
                    self.is_valid_cnt += 1

        return r



    def _increment(self):
        self.curr += 1
        if self.curr < self.__len__():
            return self.events[self.curr]['state']
        else:
            return None

    def reset(self):
        self.curr = 0
        event = self.__getitem__(self.curr)
        self._increment()
        return event['state']

    def __len__(self):
        return len(self.events) - 1

    def __iter__(self):
        return self

    def __getitem__(self, i):
        return self.events[i]


    """def __next__(self):
        self.curr += 1
        if self.curr < self.max:
            return self.__getitem__(self.curr)
        raise StopIteration"""



if __name__ == "__main__":
    df = pd.read_csv("./example-display.csv")
    df['last_scanned_datetime'] = pd.to_datetime(df['last_scanned_datetime'])
    evaluator = OfflineDisplayPolicyEvaluator(df)
    policy = RandomPolicy(
        products=list(df['name'].unique()),
        range=(2, 8)
    )
    n_step = len(evaluator)
    s = evaluator.reset()
    total_reward = 0
    for i in range(n_step):
        a = policy(s)
        r, s, true_payoffs = evaluator.step(a=a)
        print(s)
        total_reward += r
    print('total: ', total_reward / evaluator.is_valid_cnt)