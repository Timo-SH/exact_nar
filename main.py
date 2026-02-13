import torch
import numpy as np
import torch_geometric
import argparse
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import copy
import os
import pandas as pd
from torch.distributed import destroy_process_group
import wandb

from utils import setup_run, ddp_setup, accelerator_setup, log_metrics, metric_dict_to_str, RuntimeMemProfile
from model import load_model
from graphgen import load_train_data, load_test_data, load_complete_test_data, load_random_train_data, load_random_train_data_paths





def main():
    """parameter setup for model and tasks"""
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--seed", type=int, default=42, help="Sets the random seed used across the complete training.")
    parser.add_argument("--run_name", type=str, default="default_run", help="A name for this specific run.")

    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_steps", type=int, default=100000, help="Setting the number of steps a model runs for.")
    parser.add_argument("--warmup_iters", type=int, default=1000, help="The number of warm up iterations for the learning rate scheduler.")
    parser.add_argument("--gradient_norm", type=float, default=10.0, help="The gradient norm for gradient updates in the backward pass.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay for the optimizer.")
    parser.add_argument("--grad_accum", type=int, default=1, help="Number of gradient accumulation steps to perform before each optimizer step.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--weight", type=float, default=5.0, help="Weight assigned to edges in the training dataset.")
    parser.add_argument("--train_loss", type=str, default="l1_regularized", choices=["l1_regularized", "mae"], help="Training loss function.")
    parser.add_argument("--eval_loss", type=str, default="relative_mae", choices=["relative_mae", "mae"], help="Evaluation loss function.")
    parser.add_argument("--loss_eta", type=float, default=0.1, help="Regularization coefficient for l1_regularized loss.")
    parser.add_argument("--reg_term_type", type=str, default="custom_reg_term", choices=["reg_term", "custom_reg_term"], help="Regularization term used by l1_regularized loss.")
    parser.add_argument("--reg_power", type=float, default=1.0, help="Power p used in selected regularization term.")

    # IO
    parser.add_argument("--log_every", type=int, default=1000, help="determines the interval in steps for which results should be logged to a .txt file.")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--test_during_training", action="store_true", help="Sets the option to provide results on the test set during training. If not set, the test result will only be obtained at the end of training")
    parser.add_argument("--checkpoint", type=str, default=None, help="gives an absolute or relative path to a checkpoint file to be used for this run.")
    parser.add_argument("--resource_management", action="store_true", help="enables logging of resource usage (GPU and RAM) across 1000 steps. This option disables the normal training")
    parser.add_argument("--save_every", type=int, default=10000, help="determines the interval in steps for which a model checkpoint should be saved.")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--path", type=str, default=None, help="Optional CSV path to append final results.")
    parser.add_argument("--train_dataset", type=str, default="random_paths", choices=["line", "random", "random_paths"], help="Training dataset to load.")
    parser.add_argument("--test_dataset", type=str, default="complete", choices=["er", "complete"], help="Test dataset to load.")
    parser.add_argument("--less_expressive_mpnn", type=str, default="auto", choices=["auto", "true", "false"], help="Controls less_expressive_mpnn dataset generation mode.")
    

    #GNN
    parser.add_argument("--input_dim", type=int, default=1, help="Dimension of input node features.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Dimension of hidden layers in the GNN.")
    parser.add_argument("--output_dim", type=int, default=1, help="Dimension of output node features.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers.")
    parser.add_argument("--num_layers_mlp", type=int, default=2, help="Number of layers in the MLPs used within the GNN.")
    #TODO: Add model specific arguments
    parser.add_argument("--num_graphs", type=int, default=200, help="Number of graphs in the test dataset.")
    parser.add_argument("--num_nodes_test", type=int, default=1024, help="Number of nodes in each graph in the test dataset.")
    parser.add_argument("--steps_test", type=int, default=2, help="Number of Bellman-Ford steps in each graph in the test dataset.")

    args = parser.parse_args()
    if args.seed is not None:
        torch_geometric.seed_everything(args.seed)

    ddp_setup()
    ctx, _, device = setup_run()
    device, device_id, device_count, main_process = accelerator_setup()
    print(f"CUDA version: {torch.version.cuda}")

    less_expressive_override = {"auto": None, "true": True, "false": False}[args.less_expressive_mpnn]

    train_loader_fns = {
        "line": load_train_data,
        "random": load_random_train_data,
        "random_paths": load_random_train_data_paths,
    }
    test_loader_fns = {
        "er": load_test_data,
        "complete": load_complete_test_data,
    }

    train_loader_kwargs = {"batch_size": args.batch_size, "weight": args.weight}
    test_loader_kwargs = {
        "num_graphs_er": args.num_graphs,
        "num_nodes": args.num_nodes_test,
        "steps": args.steps_test,
    }
    if less_expressive_override is not None:
        train_loader_kwargs["less_expressive_mpnn"] = less_expressive_override
        test_loader_kwargs["less_expressive_mpnn"] = less_expressive_override

    # Prepare data and model
    data = train_loader_fns[args.train_dataset](**train_loader_kwargs)
    test_data = test_loader_fns[args.test_dataset](**test_loader_kwargs)
    orig_model = load_model(args)

    logger.info(orig_model)
    orig_model.reset_parameters()
    model = orig_model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {param_count}")

    if device_count > 1:
        logger.info("Creating DDP module")
        model = DDP(model, device_ids=[device_id])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Define loss helpers and evaluation before training loop
    def _base_model(model):
        return model.module if isinstance(model, DDP) else model

    def reg_term(model, p):
        reg = 0.0
        model_to_regularize = _base_model(model)
        for param in model_to_regularize.parameters():
            reg += torch.pow(param.abs(), p).sum()
        return reg
    

    def custom_reg_term(model, p):
        model_to_regularize = _base_model(model)
        reg = 0.0
        for i, gnn_layer in enumerate(model_to_regularize.module_list):
            K = i + 1
            first_linear_layer = False
            for layer in gnn_layer.aggr_mlp:
                if isinstance(layer, torch.nn.Linear):
                    if layer.bias is not None:
                        reg += torch.pow(layer.bias.abs(), p).sum()
                    if not first_linear_layer:
                        first_linear_layer = True
                        W_x = layer.weight[:, :1]
                        W_edge = layer.weight[:, 1:]
                        reg += torch.pow(W_x.abs(), p).sum() * K 
                        reg += torch.pow(W_edge.abs(), p).sum() 
                    else:
                        reg += torch.pow(layer.weight.abs(), p).sum() * K

            for layer in gnn_layer.update_mlp:
                if isinstance(layer, torch.nn.Linear):
                    if layer.bias is not None:
                        reg += torch.pow(layer.bias.abs(), p).sum()
                    reg += torch.pow(layer.weight.abs(), p).sum() * (K+1)
            
        return reg
            

    def _select_scalar(out, batch):
        return out, batch.y.view(-1, 1) if batch.y.dim() == 1 else batch.y

    reg_term_fns = {
        "reg_term": reg_term,
        "custom_reg_term": custom_reg_term,
    }
    selected_reg_term_fn = reg_term_fns[args.reg_term_type]

    def l1_regularized_loss(out, batch, model=None, eta=0.1, regularizer=None, p=1.0):
        pred, target = _select_scalar(out, batch)
        return torch.mean(torch.abs(pred - target)), eta * regularizer(model, p)

    def mean_absolute_error(out, batch, **kwargs):
        pred, target = _select_scalar(out, batch)
        return torch.mean(torch.abs(pred - target)), 0.0
    
    def test_error(out, batch, **kwargs):
        pred, target = _select_scalar(out, batch)
        return torch.mean(torch.abs(pred-target)/(target + 1)), 0.0

    train_loss_fns = {
        "l1_regularized": lambda out, batch: l1_regularized_loss(
            out,
            batch,
            model=model,
            eta=args.loss_eta,
            regularizer=selected_reg_term_fn,
            p=args.reg_power,
        ),
        "mae": lambda out, batch: mean_absolute_error(out, batch),
    }
    eval_loss_fns = {
        "relative_mae": lambda out, batch: test_error(out, batch),
        "mae": lambda out, batch: mean_absolute_error(out, batch),
    }

    selected_train_loss_fn = train_loss_fns[args.train_loss]
    selected_eval_loss_fn = eval_loss_fns[args.eval_loss]

    def get_loss():
        batch = data.sample().to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr)

        loss, reg_loss = selected_train_loss_fn(out, batch)
        return loss, reg_loss

    def get_eval_loss(batch):
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        loss, reg_loss = selected_eval_loss_fn(out, batch)
        return loss, reg_loss

    def load_evaluation():
        def evaluate(dloader, model, ctx, device):
            model.eval()
            total_loss = 0.0
            count = 0
            with torch.no_grad(), ctx:
                for batch in dloader:
                    batch = batch.to(device)
                    loss, _ = get_eval_loss(batch)
                    total_loss += loss.item() * getattr(batch, "num_graphs", 1)
                    count += getattr(batch, "num_graphs", 1)
            return {"mae": total_loss / max(count, 1)}
        return evaluate

    evaluate = load_evaluation()

    logger.info("Starting training 🍿")
    warmup_iters = args.warmup_iters

    logger.info("Using constant learning rate across all steps.")

    best_loss = None
    test_metrics = {}
    state_dict = {}
    best_model = None

    if args.resource_management:
        Runtime_calc = RuntimeMemProfile()
        args.log_every = 1e9
        args.num_steps = 100

    wandb.init(project="exact_ar", config=vars(args), name=args.run_name) 
    wandb.watch(orig_model, log="parameters", log_freq=100)
    loss_window = []
    loss_reg_window = []
    loss_train_window = []
    os.makedirs(args.save_dir, exist_ok=True)

    for step in range(args.num_steps):

        # NOTE: Step 1 to allow for compile/warmup
        if step == 1 and args.resource_management:
            Runtime_calc.activate()

        _loss = 0.0
        _train_loss = 0.0
        _reg_loss = 0.0
        for _ in range(args.grad_accum):
            train_loss, reg_loss = get_loss()
            total_loss = (train_loss + reg_loss) / args.grad_accum
            total_loss.backward()
            _loss += total_loss.item()
            _train_loss += train_loss.item() / args.grad_accum
            _reg_loss += reg_loss.item() / args.grad_accum

        nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        

        if main_process:
            loss_window.append(_loss)
            loss_reg_window.append(_reg_loss)
            loss_train_window.append(_train_loss)
            if (step + 1) % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                loss = sum(loss_window) / len(loss_window)
                loss_reg = sum(loss_reg_window) / len(loss_reg_window)
                loss_train = sum(loss_train_window) / len(loss_train_window)
                log_metrics(step, lr, loss, best_loss, {}, {})
                wandb.log({
                    "train/loss": loss,
                    "train/lr": lr, 
                    "train/train_loss":loss_train,
                    "train/reg_loss":loss_reg})
                loss_window = []
                loss_reg_window = []
                loss_train_window = []
                if args.test_during_training:
                    test_metrics = evaluate(test_data, model, ctx, device)
                    wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
                    model.train()

        if main_process and (step + 1) % args.save_every == 0:
            path = os.path.join(args.save_dir, f"checkpoint_step_{step+1}.pt")
            to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save({
                "gnn_state_dict": to_save,
                "optimizer_state_dict": optimizer.state_dict()
            }, path)
            model.train()

    if args.resource_management:
        Runtime_calc.stop(step - 1)

    logger.info("Training complete ✨")

    path = os.path.join(args.save_dir, f"final_model.pt")
    to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save({"gnn_state_dict": to_save}, path)

    if not args.resource_management:
        if not args.test_during_training:
            eval_model = best_model if best_model is not None else model
            test_metrics = evaluate(test_data, eval_model, ctx, device)

        logger.info(f"Final results: {metric_dict_to_str(test_metrics)}")
        
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
        if args.path is not None:
            results = [
                {
                    "best_val_score": best_loss,
                    **test_metrics,
                    **vars(args),
                }
            ]
            logger.info(f"Logging results to {args.path}")
            if os.path.exists(path := args.path):
                pd.DataFrame(results).to_csv(path, header=False, mode="a", index=False)
            else:
                pd.DataFrame(results).to_csv(path, header=True, index=False)

    if device_count > 1:
        destroy_process_group()

if __name__ == "__main__":
    main()


