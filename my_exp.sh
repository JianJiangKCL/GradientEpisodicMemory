#!/bin/bash

MY_PYTHON="python"
MNIST_ROTA="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 1000 --data_file mnist_rotations.pt    --cuda yes  --seed 0"
MNIST_PERM="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 1000 --data_file mnist_permutations.pt --cuda yes  --seed 0"
CIFAR_100i="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path results/ --batch_size 10 --log_every 100 --samples_per_task 2500 --data_file cifar100.pt           --cuda yes --seed 0"

# cd data/
# cd raw/

# $MY_PYTHON raw.py
# model "GEM"


# cd data/
# $MY_PYTHON mnist_permutations.py \
# 	--o mnist_permutations.pt \
# 	--seed 0 \
# 	--n_tasks 20



# $MY_PYTHON mnist_rotations.py \
# 	--o mnist_rotations.pt\
# 	--seed 0 \
# 	--min_rot 0 \
# 	--max_rot 180 \
# 	--n_tasks 20

# $MY_PYTHON cifar100.py \
# 	--o cifar100.pt \
# 	--seed 0 \
# 	--n_tasks 20

$MY_PYTHON main.py $MNIST_ROTA --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5 --sampling_rate 0.8
#$MY_PYTHON main.py $MNIST_PERM --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5
#$MY_PYTHON main.py $CIFAR_100i --model gem --lr 0.1 --n_memories 256 --memory_strength 0.5

# plot results
cd results/
$MY_PYTHON plot_results.py
cd ..
