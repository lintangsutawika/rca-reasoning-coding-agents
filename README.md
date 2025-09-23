# Reasoning Coding Agents

## Training

1. Training long horizon RL: Train an agent on a full sequence
2. Training thinking RL: Train an agent to reason over partial trajectories to predict the next action.

## Eval

```
uv run rca/run.py --subset smith \
    --split train \
    --slice :10 \
    --output runs/ \
    --config config_yaml/swebench.yaml \
    --model-class litellm \
    --model litellm_proxy/neulab/claude-sonnet-4-20250514
```