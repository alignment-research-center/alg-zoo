import functools
import os
import time
from contextlib import nullcontext

import blobfile as bf
import torch as th

from .architectures import (
    DistMLP,
    DistRNN,
    DistTransformer,
    ScalarMLP,
    ScalarRNN,
    ScalarTransformer,
)
from .logger import Logger
from .tasks import task_registry


def torch_device():
    gpu_offset = os.getenv("GPU_OFFSET", "0")
    if th.cuda.is_available():
        return th.device(f"cuda:{gpu_offset}")
    else:
        return th.device("cpu")


def lr_mult_schedule(step, *, n_steps, warmup_steps, decay_min_mult):
    warmup = min(step / warmup_steps, 1)
    decay = max(decay_min_mult ** (2 * step / n_steps), decay_min_mult)
    return warmup * decay


def save_model(model, model_name):
    for save_dir_env_var in ["RESULTS_DIR", "RESULTS_BLOB_DIR"]:
        save_dir = os.getenv(save_dir_env_var)
        if save_dir is None:
            if "BLOB" in save_dir_env_var:
                continue
            else:
                save_dir = os.getcwd()
        save_filename = f"{model_name}.pth"
        save_file_path = os.path.join(save_dir, save_filename)
        print(f"Saving model to: {save_file_path}")
        with bf.BlobFile(save_file_path, "wb") as save_file:
            th.save(model.state_dict(), save_file)


def train(
    task_name,
    *,
    hidden_size,
    seq_len,
    n_transformer_layers=2,
    n_transformer_heads=1,
    n_mlp_hidden_layers=None,
    bias=False,
    discrete_eot=False,
    weight_decay=0.001,
    lr_base=0.04,
    lr_warmup_steps=2**14,
    lr_decay_min_mult=0.01,
    batch_size=1024,
    n_train=2**30,
    n_val=2**24,
    seed=0,
    checkpoints=10,
    log_every=1024,
):
    """
    Train a small network to perform an algorithmic task.

    Arguments:
    - task_name: name of task to look up in task_registry
    - hidden_size: size of model's hidden layers
    - seq_len: length of each input sequence
    - n_transformer_layers: for discrete inputs, number of transformer layers
    - n_transformer_heads: for discrete inputs, number of transformer heads
    - n_mlp_hidden_layers: for continuous inputs, number of hidden MLP layers,
        or None to use an RNN
    - bias: whether to use biases in the model
    - discrete_eot: for discrete inputs, whether to append an extra token to
        each sequence for the transformer to use to produce its output
    - weight_decay: weight decay parameter for AdamW
    - lr_base: base learning rate, gets divided by sqrt(hidden_size)
    - lr_warmup_steps: number of gradient steps to warm up learning rate over
    - lr_decay_min_mult: minimum multiple of original learning rate at which to
        stop exponential decay
    - batch_size: number of sequences in each minibatch
    - n_train: total number of sequences used for training
    - n_val: total number of sequences used for validation
    - seed: manual torch seed
    - checkpoints: number of intermediate checkpoints to save, spaced out by
        factors of 2
    - log_every: log information every this number of gradient steps
    """
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    task, input_type, task_type = task_registry[task_name]
    if input_type == "continuous" and n_mlp_hidden_layers is not None:
        use_mlp = True
        task_label = f"{task_name}_mlp"
    else:
        use_mlp = False
        task_label = task_name
    transformer_kwargs = {
        "n_layers": n_transformer_layers,
        "n_heads": n_transformer_heads,
    }
    mlp_kwargs = {
        "n_hidden_layers": n_mlp_hidden_layers,
    }
    model_cls, model_kwargs = {
        ("continuous", "dist", False): (DistRNN, {}),
        ("continuous", "scalar", False): (ScalarRNN, {}),
        ("discrete", "dist", False): (DistTransformer, transformer_kwargs),
        ("discrete", "scalar", False): (ScalarTransformer, transformer_kwargs),
        ("continuous", "dist", True): (DistMLP, mlp_kwargs),
        ("continuous", "scalar", True): (ScalarMLP, mlp_kwargs),
    }[(input_type, task_type, use_mlp)]
    model = model_cls(
        hidden_size=hidden_size, seq_len=seq_len, bias=bias, **model_kwargs
    )
    model = model.to(torch_device())
    save_model(model, f"{task_label}_{hidden_size}_{seq_len}_{0}_{seed}")
    lr = lr_base / hidden_size**0.5
    optimizer = th.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_steps = {"train": n_train // batch_size, "val": n_val // batch_size}
    lr_lambda = functools.partial(
        lr_mult_schedule,
        n_steps=n_steps["train"],
        warmup_steps=lr_warmup_steps,
        decay_min_mult=lr_decay_min_mult,
    )
    lr_scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    checkpoint_steps = [n_steps["train"] // 2**i for i in range(checkpoints + 1)]
    for period in ["train", "val"]:
        print(f"Running {period}:")
        logger = Logger(
            f"{period}_{task_label}_{hidden_size}_{seq_len}_{n_train}_{seed}"
        )
        start_time = time.time()
        with th.no_grad() if period != "train" else nullcontext():
            for step in range(n_steps[period]):
                if input_type == "continuous":
                    x = th.randn(
                        batch_size, seq_len, dtype=model.dtype, device=model.device
                    )
                elif input_type == "discrete":
                    x = th.randint(
                        low=0,
                        high=seq_len - (1 if discrete_eot else 0),
                        size=(batch_size, seq_len - (1 if discrete_eot else 0)),
                        device=model.device,
                    )
                    if discrete_eot:
                        x = th.cat((x, th.ones_like(x[:, :1]) * (seq_len - 1)), dim=1)
                else:
                    raise ValueError(f"Unrecognized input_type {input_type}")
                targets = task(x)
                if task_type == "dist":
                    logits = model.forward(x)
                    loss = th.nn.functional.cross_entropy(logits, targets)
                    accuracy = (logits.argmax(dim=-1) == targets).float().mean()
                elif task_type == "scalar":
                    outputs = model.forward(x).squeeze(-1)
                    loss = ((outputs - targets) ** 2).mean()
                else:
                    raise ValueError(f"Unrecognized task_type {task_type}")
                if period == "train":
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                logger.stage("step", step + 1, "max")
                logger.stage("seconds", time.time() - start_time, "max")
                logger.stage("loss", loss.item(), "mean")
                if task_type == "dist":
                    logger.stage("accuracy", accuracy.item(), "mean")
                logger.stage("lr", optimizer.param_groups[0]["lr"], "mean")
                if (step + 1) % log_every == 0:
                    logger.log()
                if period == "train" and (step + 1) in checkpoint_steps:
                    save_model(
                        model,
                        f"{task_label}_{hidden_size}_{seq_len}_{(step + 1) * batch_size}_{seed}",
                    )
