from typing import Tuple, List, Dict
import random
import numpy as np


class CandidateGenerator(object):
    # Adjacency List
    # The adjacency list is a nested dictionary that describes the relationships between products.
    # The keys of the parent dictionary include all the product id's that have appeared in a "Cooler" Display
    # Audit with a "Coke" manufactured product in the past 30 days. Each value is a dictionary where the
    # keys are other products that have appeared with that product and are the same category and the values
    # are the number of times the two products have appeared together.

    # Example:
    #
    # adj_list = {
    #   "Diet-Coke-20oz-uuid":
    #       {
    #           "Coke-20oz-uuid": 250
    #           "Coke-Zero-20oz-uuid": 235
    #           "Sprite-can-uuid": 110
    #       },
    #    "Monster Sunsrise":
    #       {
    #           "Monster Ultra": 5
    #       }
    # }
    #
    #
    # In the above example, Diet Coke 20oz has shown up on the same cooler as Coke 250 times, Coke Zero 235
    # times and Sprite 110 times. Monster Sunrise and Monster Ultra have been on the same Cooler 5
    # times.

    def gen(self, adj_list: dict, seed: List) -> Tuple[List, Dict]:
        # TODO: Resolve mismatch between product names and ID's in sampler
        candidate_dict = self.pre_sampled_candidate_set(seed, adj_list)

        #pruned_candidate_dict = {}
        #for id, value in candidate_dict.items():
            #if id in acceptable_products:
            #    pruned_candidate_dict[id] = value

        #self.logger.info("Probability Dictionary")

        return candidate_dict

    # Pre-Sampling vs. Post-Sampling
    #
    # The candidate set is generated using the adjacency list and a "seed" array of products. This is typically
    # the set of products currently on the display. We then use one of two methods to generate the set: Pre-Sampling
    # or Post-Sampling.
    #
    # When pre-sampling, the child dictionary for each "seed" product is retrieved and sampled n times, using the
    # values of the dictions as weights. After n samples have been taken from each seed dictionary, the samples are
    # aggregated, with each appearance of a particular product receiving one "vote". This "vote" dictionary is
    # returned with the number of votes each product received as the weights.
    #
    # When post-sampling, the dictionaries for each of the seed products are first aggregated, with the weights of the
    # same product being added together. After aggregation the set is then sampled n times, and returned.
    #
    # Key Difference: When Pre-Sampling we sample without replacement. One product can only get a
    # maximum of 1 vote from each seed product set. This likely leads to a higher diversity in our candidate set
    # because it prevents products with extremely high weights from consistently dominating the candidate set.

    def pre_sampled_candidate_set(self, ids, data, num_samples=20):
        vote_dict = {}
        for seed_id in ids:
            if seed_id not in data:
                continue
            upc_dict = data[seed_id]

            # Filter out the products already in the seed set
            pruned_dict = {}
            for key, val in upc_dict.items():
                if key not in ids:
                    pruned_dict[key] = val

            # If there are no new candidates, continue
            if len(pruned_dict.keys()) == 0:
                continue

            upc_weights = np.array(list(pruned_dict.values())) / np.array(list(pruned_dict.values())).sum()
            upc_samples = list(pruned_dict.keys())
            choices = np.random.choice(upc_samples, min([num_samples, len(upc_samples)]), p=upc_weights, replace=False)

            for choice in choices:
                if choice in vote_dict:
                    vote_dict[choice] += 1
                else:
                    vote_dict[choice] = 1

        vote_dict = {k: v for k, v in sorted(vote_dict.items(), key=lambda item: -item[1])}
        total = sum(vote_dict.values(), 0.0)
        prob_dict = {k: v / total for k, v in vote_dict.items()}

        return prob_dict

    def post_sampled_candidate_set(self, ids, data, num_candidates=20):
        sample_dict = {}
        for seed_id in ids:
            if seed_id not in data:
                continue
            upc_dict = data[seed_id]

            for choice in upc_dict.keys():
                if choice in ids:
                    continue
                if choice in sample_dict:
                    sample_dict[choice] += upc_dict[choice]
                else:
                    sample_dict[choice] = upc_dict[choice]

        upc_weights = np.array(list(sample_dict.values())) / np.array(list(sample_dict.values())).sum()
        upc_samples = list(sample_dict.keys())

        choices = np.random.choice(upc_samples, min([num_candidates, len(upc_samples)]), p=upc_weights, replace=False)

        pruned_cands = {}
        for key, val in sample_dict.items():
            if key in choices:
                pruned_cands[key] = val

        prob_dict = {k: v / sum(pruned_cands.values(), 0.0) for k, v in pruned_cands.items()}

        return prob_dict
