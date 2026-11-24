import argparse
import logging
import os

import numpy as np
from numpy import typing as npt
import torch
from tqdm import tqdm

from cs336_basics.transformer import AdamW, Transformer, cross_entropy, get_batch, gradient_clipping, load_checkpoint, lr_cosine_schedule, save_checkpoint

def train(args, train_dataset: npt.NDArray, valid_dataset: npt.NDArray):
    # assert args
    
    transformer = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
        dtype=torch.float32,
    )

    optimizer = AdamW(
        transformer.parameters(),
        lr=args.lr_max,
        betas=args.betas,
        weight_decay=args.weight_decay,
    )

    if args.device == "cpu":
        transformer = torch.compile(transformer)
    elif args.device.startswith("mps"):
        transformer = torch.compile(transformer, backend="aot_eager")

    iter = try_load_checkpoint(args.ckpt_path, transformer, optimizer)

    for iter in tqdm(range(iter, args.max_iterations + 1)):
        x, y = get_batch(
            dataset=train_dataset,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
        )
        logits = transformer.forward(x)

        # reshape to 2D for cross_entropy
        logits = logits.reshape(-1, args.vocab_size)
        y = y.reshape(-1)
        loss = cross_entropy(logits, y)

        lr = lr_cosine_schedule(
            t=iter,
            lr_max=args.lr_max,
            lr_min=args.lr_min,
            t_w=args.warmup_iters,
            t_c=args.cosine_annealing_iters,
        )
        for params_group in optimizer.param_groups:
            params_group["lr"] = lr

        optimizer.zero_grad()
        loss.backward()

        gradient_norm = gradient_clipping(
            params=transformer.parameters(),
            max_lr_norm=args.gradient_clipping,
        )
        optimizer.step()

        if iter % args.ckpt_interval == 0:
            save_checkpoint(
                model=transformer,
                optimizer=optimizer,
                iter=iter,
                out=args.ckpt_path,
            )
            logging.info(
                f"Checkpoint saved at iter:{iter}, loss:{loss.item():.6f}"
            )

def try_load_checkpoint(checkpoint_path, transformer, optimizer) -> int:
    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, transformer, optimizer)
    return 0

def load_data(data_path) -> npt.NDArray:
    return np.load(data_path, mmap_mode="r")

def init_args():
    parser = argparse.ArgumentParser()
    # transformer
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # data
    parser.add_argument("--train_data_path", type=str, default="tiny_stories_train.npy")
    parser.add_argument("--valid_data_path", type=str, default="tiny_stories_valid.npy")

    # optimizer
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
    parser.add_argument("--gradient_clipping", type=float, default=1.0)

    # checkpoint
    parser.add_argument(
        "--ckpt_path", type=str, default="checkpoints/checkpoint.pt"
    )
    parser.add_argument("--ckpt_interval", type=int, default=100)

    # lr schedule
    parser.add_argument("--lr_max", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--cosine_annealing_iters", type=int, default=5000)

    args = parser.parse_args()
    return args

def init_log():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

if __name__ == "__main__":
    init_log()
    args = init_args()

    train_dataset = load_data(args.train_data_path)
    valid_dataset = load_data(args.valid_data_path)
    train_dataset = train_dataset[:1000]
    valid_dataset = valid_dataset[:100]

    train(args, train_dataset, valid_dataset)
