#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research

# CIFAR100 Standard Memory Seeds
for SEED in 0
do
  /usr/bin/time -v python launch.py \
    --experiment_name "CIFAR100_MEMO1_SEED${SEED}" \
    --memo_per_class_context "50" \
    --context_layers 0 1 2 3 4 5 6 7 8 9 10 11 \
    --context_learner "LogisticRegression(random_state=0, max_iter=20, C=0.005)" \
    --dataset "CIFAR100" \
    --number_of_tasks "5" \
    --model "VGG11_SLIM" \
    --seed "${SEED}" \
    --learning_rate "0.005" \
    --batch_size "32" \
    --weight_decay "0.001" \
    --phase_epochs "5" \
    --activation_perc "95.0" \
    --max_phases "5" \
    --use_wandb \
    --wandb_project "SYNAPSE_CIFAR100_${SEED}" \
    >> CIFAR100_MEMO1_SEED${SEED}.txt
done


# # TinyImageNet Standard Memory Seeds
# for SEED in 0
# do
#   /usr/bin/time -v python launch.py \
#     --experiment_name "TinyImagenet_MEMO1_SEED${SEED}" \
#     --memo_per_class_context "25" \
#     --context_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 \
#     --context_learner "LogisticRegression(random_state=0, max_iter=25, C=0.001)" \
#     --dataset "TinyImagenet" \
#     --number_of_tasks "5" \
#     --model "ResNet18" \
#     --seed "${SEED}" \
#     --learning_rate "0.01" \
#     --batch_size "64" \
#     --weight_decay "0.0" \
#     --phase_epochs "5" \
#     --activation_perc "97.5" \
#     --max_phases "5" \
#     --use_wandb \
#     --wandb_project "SYNAPSE_TinyImagenet_${SEED}" \
#     >> TinyImagenet_MEMO1_SEED${SEED}.txt
# done

