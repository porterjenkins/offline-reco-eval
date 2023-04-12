import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List
from tqdm import tqdm

from policies import RandomPolicy


class DisplayState(object):

    def __init__(self, prod_quantity: Optional[dict] = None, max_slots: Optional[int] = None, timestamp: Optional[datetime] = None):
        self.max_slots = max_slots
        self.ts = timestamp
        if prod_quantity is not None:
            self.prods = set(prod_quantity.keys())
        self.quantities = prod_quantity


    def set_time(self, ts: datetime):
        self.ts = ts

    def set_max_slots(self, max_slots: int):
        self.max_slots = max_slots

    def __str__(self):
        return str(self.ts) + ": " + str(self.prods)



class OfflineDisplayPolicyEvaluator(object):

    """
    Li et al. 2011
    'Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms
    """

    def __init__(self, disp_name: str, df: pd.DataFrame):
        self.disp_name = disp_name
        self.events = self.get_events(df)
        self.curr = 0
        self.is_valid_cnt = 0

    @staticmethod
    def get_events(df: pd.DataFrame):
        events = []
        groups = df[['last_scanned_datetime', 'name', 'previous_post_scan_num_facings', 'payoff', 'max_slots']].groupby('last_scanned_datetime')
        state = DisplayState() # empty state
        i = 0
        for k, v in groups:

            state.set_time(v['last_scanned_datetime'].max())
            state.set_max_slots(v['max_slots'].max())

            action = dict(zip(v['name'], v['previous_post_scan_num_facings']))
            payoff = dict(zip(v['name'], v['payoff']))
            if i > 0:
                # skip initial action (no prev state available)
                events.append(
                    {'state': state, 'payoff': payoff, "action": action, "timestamp": k}
                )

            # create new state: state at t+1 is equal to action at t
            state = DisplayState(
                prod_quantity=action
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


class OfflineEvaluator(object):

    def __init__(self, display_eval: List[OfflineDisplayPolicyEvaluator], num_iter:int =10):

        self.display_evals = display_eval
        self.num_iter = num_iter
    def main(self, policy):
        n_disp = len(self.display_evals)
        print(f"Beginning Eval: {n_disp} Displays")
        reward_mtx = np.zeros((n_disp, self.num_iter))
        for i, disp_evaluator in enumerate(self.display_evals):
            print(f"Beginning Display: {disp_evaluator.disp_name}")
            for j in tqdm(range(self.num_iter)):
                n_step = len(disp_evaluator)
                s = disp_evaluator.reset()
                total_reward = 0
                for k in range(n_step):
                    a = policy(s)
                    r, s, true_payoffs = disp_evaluator.step(a=a)
                    total_reward += r
                if disp_evaluator.is_valid_cnt > 0:
                    # avoid division by zero
                    total_reward = total_reward / disp_evaluator.is_valid_cnt

                reward_mtx[i, j] = total_reward

        print("means:", np.mean(reward_mtx, axis=1))
        print("stds", np.std(reward_mtx, axis=1))

        print("global mean: ", np.mean(reward_mtx))
        print("global std: ", np.std(reward_mtx))

    @classmethod
    def build_from_csv(cls, fpath: str, iter: int):
        df = pd.read_csv(fpath)
        df['last_scanned_datetime'] = pd.to_datetime(df['last_scanned_datetime'])
        df.sort_values(by=['display_id', 'last_scanned_datetime'], ascending=False, inplace=True)
        disp_ids = df['display_id'].unique()
        disp_evals = []
        for d_id in disp_ids:
            disp_data = df[df['display_id'] == d_id]
            disp_evals.append(
                OfflineDisplayPolicyEvaluator(disp_name=d_id, df=disp_data)
            )

        evaluator = OfflineEvaluator(display_eval=disp_evals, num_iter=iter)

        return evaluator



if __name__ == "__main__":
    df = pd.read_csv("./data/example-display.csv")
    df['last_scanned_datetime'] = pd.to_datetime(df['last_scanned_datetime'])
    evaluator = OfflineDisplayPolicyEvaluator(disp_name=df['display_id'].unique()[0], df=df)
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
        total_reward += r
    print('total: ', total_reward / evaluator.is_valid_cnt)