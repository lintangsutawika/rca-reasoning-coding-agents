from rca.rewards import reward

@reward("resolve_reward")
def resolve_reward(
    patch: str,
    instance: Dict,
    **kwargs
    ):

    if patch.startswith("diff --git"):
        reward = 1
    else:
        reward = 0