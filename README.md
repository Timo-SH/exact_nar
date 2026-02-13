
# Code Repository

The following code repository provides the source code for the paper "Which Algorithms Can Graph Neural Networks Learn?".

## Requirements

The code requires the following packages:

- torch
- torch-geometric
- networkx
- tqdm
- numpy
- pandas
- wandb
- argparse
- loguru

## CLI options (main.py)

The main.py script gives the following options to replicate experiments made in the paper: 

### Dataset selection

- `--train_dataset {line,random,random_paths}`
        - `line`: standard path/line training dataset
        - `random`: randomly generated training graphs
        - `random_paths`: randomly generated training graphs + required path graphs
- `--test_dataset {er,complete}`
        - `er`: ER-constdeg and ER style test dataset
        - `complete`: general test dataset (ER, complete, SBM, star, multi-line)
- `--less_expressive_mpnn {auto,true,false}`
        - `auto`: keep loader default behavior
        - `true`/`false`: force the setting for both train and test loaders

### Loss and regularization selection

- `--train_loss {l1_regularized,mae}`
- `--eval_loss {relative_mae,mae}`
- `--loss_eta` regularization coefficient for `l1_regularized`
- `--reg_term_type {reg_term,custom_reg_term}` regularizer used by `l1_regularized` where custom_reg_term denotes the regularization used in theoretical insights on Bellman-Ford
- `--reg_power FLOAT` power `p` in the regularization term 

## Experiments

Base command template:

```bash
python main.py \
        --num_steps 160000 \
        --seed 1 \
        --learning_rate 0.0001 \
        --input_dim 1 \
        --hidden_dim 64 \
        --output_dim 16 \
        --num_layers 2 \
        --num_layers_mlp 2 \
        --log_every 100 \
        --save_every 40000 \
        --run_name test \
        --batch_size 1 \
        --test_during_training \
        --weight 50 \
        --num_nodes_test 64 \
        --num_graphs 50 \
        --steps_test 2
```

### Q1 (standard setting)

```bash
python main.py \
        --num_steps 160000 \
        --seed 1 \
        --learning_rate 0.0001 \
        --input_dim 1 \
        --hidden_dim 64 \
        --output_dim 16 \
        --num_layers 2 \
        --num_layers_mlp 2 \
        --log_every 100 \
        --save_every 40000 \
        --run_name test \
        --batch_size 1 \
        --test_during_training \
        --weight 50 \
        --num_nodes_test 64 \
        --num_graphs 50 \
        --steps_test 2 \
        --train_dataset line \
        --test_dataset complete \
        --less_expressive_mpnn false \
        --train_loss l1_regularized \
        --eval_loss relative_mae \
        --reg_term_type custom_reg_term \
        --reg_power 1
```

### Q2 (less expressive MPNN features)


```bash
python main.py \
        --num_steps 160000 \
        --seed 1 \
        --learning_rate 0.0001 \
        --input_dim 1 \
        --hidden_dim 64 \
        --output_dim 16 \
        --num_layers 2 \
        --num_layers_mlp 2 \
        --log_every 100 \
        --save_every 40000 \
        --run_name test \
        --batch_size 1 \
        --test_during_training \
        --weight 50 \
        --num_nodes_test 64 \
        --num_graphs 50 \
        --steps_test 2 \
        --less_expressive_mpnn true \
        --train_dataset line \
        --test_dataset complete \
        --reg_term_type custom_reg_term \
        --reg_power 1 \
        --eval_loss relative_mae \
```

### Q3 (regularization variants)

Use `--reg_term_type` to switch regularizer and `--reg_power` to set $p$.

Examples:

```bash
# Standard reg_term with L1-regularization
python main.py ... --train_loss l1_regularized --reg_term_type reg_term --reg_power 1

# Standard reg_term with L2-regularization
python main.py ... --train_loss l1_regularized --reg_term_type reg_term --reg_power 2

# Custom regularizer with p = 1
python main.py ... --train_loss l1_regularized --reg_term_type custom_reg_term --reg_power 1
```