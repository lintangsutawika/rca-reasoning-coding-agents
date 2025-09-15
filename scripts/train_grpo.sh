#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=preempt
#SBATCH --gres=gpu:L40S:4
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=512G
#SBATCH --exclude=babel-9-3,babel-4-25,babel-14-29,babel-12-9,babel-13-1,babel-10-9
##SBATCH --constraint=nvlink

. ./emac/config/.env

# sbatch emac/scripts/train_grpo.sh \
#   -m Qwen/Qwen3-4B \
#   -t openhands \
#   -d data/ \

while getopts ":m:l:n:t:d:s:f:r:v:g:e:" opt; do
  case ${opt} in
    m ) MODEL=$OPTARG;;
    l ) LANGUAGE=$OPTARG;;
    n ) N_ROLLOUTS=$OPTARG;;
    t ) TASK=$OPTARG;;
    d ) DATA_PATH=$OPTARG;;
    s ) SAVE_MODEL_PATH=$OPTARG;;
    f ) FUNCTION_NAME=$OPTARG;;
    r ) RUN_NAME=$OPTARG;;
    v ) VALIDATION_DATA_PATH=$OPTARG;;
    g ) USE_GCS=$OPTARG;;
    e ) SOURCE_TYPE=$OPTARG;;
    # \? ) echo "Usage: cmd [-u] [-p]";;
  esac
done

MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')
# Get number of GPUs available
NUM_GPUS=$(nvidia-smi -L | wc -l)
USE_GCS="${USE_GCS:-False}"
N_ROLLOUTS="${N_ROLLOUTS:-8}"
FUNCTION_NAME="${FUNCTION_NAME:-compute_score}"
MAX_LENGTH=8192
RUN_NAME="${RUN_NAME:-grpo}"
RUN_NAME=${RUN_NAME}--${MODEL_ALIAS}--${TASK}
FULL_DATA_PATH=${DATA_PATH}${TASK}
FULL_SAVE_PATH=${SAVE_MODEL_PATH}${RUN_NAME}
LOGPROB_BS=16
PPO_BS=16

python -m verl.trainer.main_ppo \
    +trainer.wandb.language=${LANGUAGE} \
    +trainer.wandb.task=${TASK} \
    +trainer.use_gcs=${USE_GCS} \
    +trainer.gcs_project=${GCS_PROJECT} \
    +trainer.gcs_token=${GCS_TOKEN} \
    +trainer.gcs_path=${GCS_PATH}${RUN_NAME} \
    trainer.validation_data_dir=${FULL_SAVE_PATH}/evaluations/ \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${FULL_DATA_PATH}/train.parquet \
    data.val_files=${FULL_DATA_PATH}/test.parquet \
    data.prompt_key=input \
    data.train_batch_size=32 \
    data.max_prompt_length=16384 \
    data.max_response_length=${MAX_LENGTH} \
    data.shuffle=True \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-sum-norm" \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${N_ROLLOUTS} \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_BS} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOGPROB_BS} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rca' \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.balance_batch=False \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.total_epochs=20 \
    trainer.total_training_steps=250 \
    trainer.default_local_dir=${FULL_SAVE_PATH} \
    custom_reward_function.path=emac/emac/rewards/reward_fn.py \
    custom_reward_function.name=${FUNCTION_NAME}
    # actor_rollout_ref.model.use_shm=True \
    # actor_rollout_ref.rollout.layered_summon=True \
    # actor_rollout_ref.actor.fsdp_config.param_offload=True \
    # actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    # actor_rollout_ref.ref.fsdp_config.param_offload=True \
    # actor_rollout_ref.ref.strategy=fsdp2 \
    # actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOGPROB_BS} \
    # actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    # actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_BS} \
    # actor_rollout_ref.actor.fsdp_config.fsdp_size=${NUM_GPUS} \
