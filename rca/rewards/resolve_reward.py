from rca.utils.timeout import timeout
from rca.rewards import reward

from rca.utils.mini_swe import (
    evaluate_trajectory,
)

@reward("resolve_reward")
def resolve_reward(
    patch,
    instance,
    **kwargs
    ):

    data_source = "swe-smith"
    reward = 0
    if patch.startswith("diff --git"):

        # with 5 minutes timeout
        try:
            with timeout(300):
                result = evaluate_trajectory(
                    instance, patch, 
                    {"environment": {"environment_class": "singularity"},
                    "cwd": "/testbed"},
                    data_source
                )
        except TimeoutError as e:
            result = None
            error = f"TimeoutError during evaluation: {e}"

        if isinstance(result, dict) and "partial_score" in result:
            reward = float(result["partial_score"])

    return reward
    