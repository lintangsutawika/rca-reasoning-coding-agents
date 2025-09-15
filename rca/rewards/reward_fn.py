import os
import re

from Levenshtein import distance, ratio

from emac.utils.parsing import parse_action

def compute_score(data_source, solution_str, ground_truth, extra_info=None):

    score_card = {
        "correct_fn": 0.0,
        "correct_param_key": 0.0,
        "correct_param_value": 0.0,
        "num_params": 0.0,
    }

    a_string = str(parse_action(solution_str, string_only=True))
    # gt_fn, gt_kwargs = parse_action(ground_truth)
    gt_string = str(parse_action(ground_truth, string_only=True))
    reward = ratio(a_string, gt_string)

    # if action_fn == gt_fn:
    #     score_card["correct_fn"] = 1.0

    # for key, value in gt_kwargs.items():
    #     score_card["num_params"] += 1.0

    #     if key in action_kwargs:
    #         score_card["correct_param_key"] += 1.0
    #         if action_kwargs[key] == value:
    #             score_card["correct_param_value"] += 1.0

    # if extra_info["reward_partial"]:
    #     reward_fn = score_card["correct_fn"]

    #     if score_card["num_params"] > 0:
    #         coeff = 0.5
    #         reward_param = score_card["correct_param_key"]
    #         reward_param += score_card["correct_param_value"]
    #         reward_param /= (score_card["num_params"] * 2)
    #     else:
    #         coeff = 1.0

    #     reward = coeff * reward_fn + coeff * reward_param
    # else:
    #     # Function and parameter must be all correct to get a reward
    #     reward = score_card["correct_fn"] * int(score_card["correct_param_value"]/score_card["num_params"])

    return {
        "data_source": data_source,
        "score": float(reward),
        "gold": gt_string,
        "action": a_string,
    }
