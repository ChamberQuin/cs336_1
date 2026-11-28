import argparse
import logging
import os

import numpy as np
from numpy import typing as npt
import torch
from tqdm import tqdm
import wandb
from jaxtyping import Float

from cs336_basics.transformer import AdamW, Transformer, cross_entropy, get_batch, gradient_clipping, load_checkpoint, lr_cosine_schedule, save_checkpoint, softmax
from cs336_basics.bpe_tokenizer import BPETokenizer

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

    # transformer = transformer
    # if args.device == "cpu":
    #     transformer = torch.compile(transformer)
    # elif args.device.startswith("mps"):
    #     transformer = torch.compile(transformer, backend="aot_eager")

    optimizer = AdamW(
        transformer.parameters(),
        lr=args.lr_max,
        betas=args.betas,
        weight_decay=args.weight_decay,
    )

    iter = try_load_checkpoint(args.ckpt_path, transformer, optimizer)

    for iter in tqdm(range(iter, args.max_iters + 1)):
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

        if args.use_wandb:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/learning_rate": lr,
                    "train/gradient_norm": gradient_norm,
                    "iter": iter,
                },
            )

        if iter % 100 == 0:
            logging.info(f"Iter:{iter} loss:{loss.item():.6f} lr:{lr:.3e} grad_norm:{gradient_norm:.3e}")

    if args.use_wandb:
        wandb.finish()

def try_load_checkpoint(checkpoint_path, transformer, optimizer) -> int:
    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, transformer, optimizer)
    return 0

def load_data(data_path) -> npt.NDArray:
    return np.load(data_path, mmap_mode="r")

def nucleus_sampling(prob: Float[torch.Tensor, " batch seq_len vocab_size"], p: float) -> int:
    assert(prob.sum() == 1)

    sorted, sorted_idx = torch.sort(prob, dim = -1, descending=True, stable=True)
    cumulative = torch.cumsum(sorted, dim = -1)

    # 保留让累计概率 >= p 的最小候选集，包括刚好超过 p 时的项
    # 不用 (cumulative - sorted_prob) <= p，避免累计概率恰好等于 p 时多留一个元素
    # 例如，prob = [0.1, 0.5, 0.25, 0.15], sorted_prob = [0.5, 0.25, 0.15, 0.1], acc_prob = [0.5, 0.75, 0.9, 1], acc_prob - sorted_prob = [0, 0.5, 0.75, 0.9]
    # 若 p = 0.5，第一个 token 实际已经满足阈值，但第二个 token 之前的累计概率也是 0.5，'<=' 会让它保留。
    mask = (cumulative - sorted) < p
    filtered = sorted[mask]
    filtered_idx = sorted_idx[mask]

    # 重新归一化，保证概率和为 1
    filtered = filtered / filtered.sum()

    sampled_idx = torch.multinomial(filtered, num_samples=1)
    return int(filtered_idx[sampled_idx].item())

@torch.no_grad()
def decoding(args, prompt):
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

    # transformer = transformer
    # if args.device == "cpu":
    #     transformer = torch.compile(transformer)
    # elif args.device.startswith("mps"):
    #     transformer = torch.compile(transformer, backend="aot_eager")

    iter = try_load_checkpoint(args.ckpt_path, transformer, None)

    end_token = "<|endoftext|>"
    tokenizer = BPETokenizer.from_files(
        args.tokenizer_path, special_tokens=[end_token]
    )
    end_token_ids = tokenizer.encode(end_token)

    encoded_prompt = tokenizer.encode(prompt)
    prompt_len = len(encoded_prompt)
    # (batch context_length) = (1, context_length)
    encoded_data = torch.zeros(1, args.context_length, dtype=torch.long, device=args.device)
    encoded_data[0, :prompt_len] = torch.tensor(encoded_prompt, dtype=torch.long, device=args.device)

    cur_idx = prompt_len
    while cur_idx < args.context_length:
        logits = transformer.forward(encoded_data)
        # 用前一个位置的 token 来预测当前位置的 token
        logits = logits[0, cur_idx - 1]
        prob = softmax(logits, dim=-1, temperature=args.temperature)
        if args.enable_nucleus_sampling:
            next_token_id = nucleus_sampling(prob, args.nucleus_sampling_p)
        else:
            next_token_id = int(torch.multinomial(prob, num_samples=1).item())
        encoded_data[0, cur_idx] = next_token_id

        if next_token_id == end_token_ids[0]:
            break
        
        cur_idx += 1

    print(tokenizer.decode(encoded_data[0, :cur_idx].cpu().tolist()))
    print(f"final idx: {cur_idx}")

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
    parser.add_argument("--train_data_path", type=str, default="/home/qinyuanbo/git/cs336_data/tiny_stories_fake.npy")
    parser.add_argument("--valid_data_path", type=str, default="/home/qinyuanbo/git/cs336_data/tiny_stories_valid.npy")

    # optimizer
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
    parser.add_argument("--gradient_clipping", type=float, default=1.0)

    # checkpoint
    parser.add_argument(
        "--ckpt_path", type=str, default="/home/qinyuanbo/git/cs336_data/checkpoints/checkpoint.pt"
    )
    parser.add_argument("--ckpt_interval", type=int, default=100)

    # lr schedule
    parser.add_argument("--lr_max", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--cosine_annealing_iters", type=int, default=5000)

    # wandb
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="llm", help="wandb 项目名称")
    parser.add_argument("--wandb_name", type=str, default=None, help="wandb 运行名称")
    parser.add_argument("--wandb_mode", type=str, default="offline")

    # decoding
    parser.add_argument("--tokenizer_path", type=str, default="/home/qinyuanbo/git/cs336_data/tiny_stories_vocab.pkl")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--enable_nucleus_sampling", action="store_true", help="是否采用 top-p 采样")
    parser.add_argument("--nucleus_sampling_p", type=float, default=0.9)

    args = parser.parse_args()
    return args

def init_log():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

if __name__ == "__main__":
    init_log()
    args = init_args()

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "d_model": args.d_model,
                "context_length": args.context_length,
                "d_model": args.d_model,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "d_ff": args.d_ff,
                "rope_theta": args.rope_theta,
                "batch_size": args.batch_size,
                "max_iters": args.max_iters,
                "weight_decay": args.weight_decay,
                "betas": args.betas,
                "gradient_clipping": args.gradient_clipping,
                "lr_max": args.lr_max,
                "lr_min": args.lr_min,
                "warmup_iters": args.warmup_iters,
                "cosine_annealing_iters": args.cosine_annealing_iters,
            },
            mode=args.wandb_mode,
        )

    # train_dataset = load_data(args.train_data_path)
    # train_dataset = load_data(args.valid_data_path)
    # valid_dataset = load_data(args.valid_data_path)
    # train_dataset = train_dataset[:1000]
    # valid_dataset = valid_dataset[:100]

    # train(args, train_dataset, valid_dataset)

    decoding(args, "You are Feynman, introduce yourself.")