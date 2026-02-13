import torch
import os
from contextlib import nullcontext
from loguru import logger
from torch.distributed import init_process_group
import time 

class RuntimeMemProfile:

    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.runs = 0
        self.max_mem = 0
        self.avg_mem = 0

    def activate(self):
        torch.cuda.synchronize()
        self.start_time = time.time()
        torch.cuda.reset_peak_memory_stats(device=None)

    def stop(self, total_steps):
        torch.cuda.synchronize()
        self.end_time = time.time()
        print("total time:" + str(self.end_time - self.start_time))
        self.avg_time = (self.end_time - self.start_time) / total_steps
        print("avg time" + str(self.avg_time))
        self.max_mem = torch.cuda.max_memory_allocated(device=None)
        print("mem allocation in MB:" + str(self.max_mem / (1024**2)))


def accelerator_setup():
    if torch.cuda.is_available():
        device = "cuda"
        device_count = torch.cuda.device_count()
        if device_count > 1:
            device_id = int(os.environ["LOCAL_RANK"])
            main_process = device_id == 0
        else:
            device_id = 0
            main_process = True
    else:
        device = "cpu"
        device_id = "cpu"
        device_count = 1
        main_process = True

    return device, device_id, device_count, main_process




def ddp_setup():
    if torch.cuda.device_count() > 1:
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))



def setup_run():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Accelerator 🚀: {device}")

    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float32"
    )
    logger.info(f"Data type: {dtype}")

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device, dtype=ptdtype)
    )

    return ctx, ptdtype, device


def metric_dict_to_str(metric_dict):
    if len(metric_dict) == 0:
        return ""

    return " | ".join([f"{name}: {score:.4f}" for name, score in metric_dict.items()])


def log_metrics(step, lr, loss, best_loss, val_metrics, test_metrics):
    val_metrics = metric_dict_to_str(val_metrics)
    test_metrics = metric_dict_to_str(test_metrics)
    best_loss_str = f"{best_loss:.4f}" if isinstance(best_loss, (int, float)) else "-"
    logger.info(
        f"Step: {step} | LR: {lr} | Train loss: {loss:.4f} | Best validation score: {best_loss_str} | {val_metrics} | {test_metrics}"
    )


