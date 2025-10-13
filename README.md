# Reasoning Coding Agents

## Training

### Dataset Preperation

Generate dataset, we will use SWE-Smith to train out models

```
uv run rca/construct_dataset.py
```

### Training Long Horizon RL

Train a model to improve performance over the full SWE task trajectory. This involves building a docker remote workspace for each prompt+rollout and calling the model checkpoint to produce the rollout before calculating the reward and advantage.

```
bash scripts/run_training.sh -m Qwen/Qwen3-4B -d data/swe_smith/
```

### Training Short Horizon RL

WIP

## Eval

WIP

## Preparation

```
git clone https://github.com/lintangsutawika/rca-reasoning-coding-agents.git
cd rca-reasoning-coding-agents

git clone https://github.com/All-Hands-AI/OpenHands.git
cp custom_openhands/system_prompt.j2 agent-sdk/openhands/sdk/agent/prompts/
cp custom_openhands/Dockerfile agent-sdk/openhands/agent_server/docker/

export $PWD/agent-sdk/
```

## Caveats that need to be solved
1. Since each docker is a remote workspace, it needs to access the VLLM outside of the docker environment. I have not been able to fix this in a way that it can access the vllm server in 0.0.0.0 or 127.0.0.1 in the host server. The work around is to use the public IP of the host server which is not ideal.
2. Not sure how to access the system message from openhands. The remote worksapce initialization includes a local workspace so the system message and user message isn't sent recived by the callback that is initialized by in the remote workspace. The workaround is to reconstruct the system message and manually add it to the collected messages.
3. There seems to be a startup error `WebSocket is not connected. Need to call \"accept\" first.` but then the rollout works just fine. Not sure what this is.