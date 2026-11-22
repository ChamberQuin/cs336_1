import os
from typing import IO, Any, BinaryIO, Iterable

from einops import einsum, rearrange
import torch
from torch import Tensor
from math import cos, pi, sin, sqrt
from jaxtyping import Float, Int, Bool
import numpy as np
from numpy import typing as npt

class Linear(torch.nn.Module):
    in_features: int
    out_features: int
    weight: torch.nn.Parameter

    def __init__(self, in_features: int, out_features: int, device = None, dtype = None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {"device": device, "dtype": dtype}
        # “感觉像行向量”，是因为数据在内存里常按 (..., in_features) 存（最后一维是特征），更像一批行向量拼起来；
        # 但为了保持线代里的 y = Wx 约定，PyTorch 选择把参数存成 (out, in)，前向实现成 y = x @ W.T + b。
        # 这样既兼容批量存储习惯，也保留了线代传统的权重形状约定。
        init_data = torch.empty((self.out_features, self.in_features), **factory_kwargs)
        torch.nn.init.trunc_normal_(
            init_data, 
            0, 
            sqrt(2 / (self.out_features + self.in_features)), 
            -3, 
            3
        )
        self.weight = torch.nn.Parameter(init_data)

    def load_weight(self, weight: Float[Tensor, "d_out d_in"]):
        state_dict = {"weight": weight}
        self.load_state_dict(state_dict)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(torch.nn.Module):
    vocab_size: int
    d_model: int
    weight: torch.nn.Parameter

    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype = None):
        super().__init__()

        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        factory_kwargs = {"device": device, "dtype": dtype}
        init_data = torch.empty((self.vocab_size, self.d_model), **factory_kwargs)
        torch.nn.init.trunc_normal_(
            init_data,
            0,
            1,
            -3,
            3
        )
        self.weight = torch.nn.Parameter(init_data)

    def load_weight(self, weight: Float[Tensor, "d_vocab d_model"]):
        state_dict = {"weight": weight}
        self.load_state_dict(state_dict)

    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        token_id_flatten = token_ids.reshape(-1)
        res = self.weight.index_select(0, token_id_flatten)
        return res.reshape((*token_ids.shape, -1))

class RMSNorm(torch.nn.Module):
    d_model: int
    eps: float
    weight: torch.nn.Parameter

    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None) -> None:
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        factory_kwargs = {"device": device, "dtype": dtype}
        init_data = torch.ones((d_model), **factory_kwargs)
        self.weight = torch.nn.Parameter(init_data)

    def load_weight(self, weight: Float[Tensor, "d_model"]):
        state_dict = {"weight": weight}
        self.load_state_dict(state_dict)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # ... d_model -> ... 1
        ## dim = -1 表示按最后一维计算 mean
        ## keep_dim = True 表示保留最为一维
        ## 两者等价于:
        ## rms = torch.sqrt(x.pow(2) / d_model + eps)
        ## rms = rms.reshape(*rms.shape, 1)
        rms = torch.sqrt(x.pow(2).mean(dim = -1, keepdim=True) + self.eps)

        result = (x / rms) * self.weight
        return result.to(in_dtype)

class SwiGLUFFN(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device = None, dtype = None):
        super().__init__()
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def load_weight(
        self, 
        w1_weight: Float[Tensor, " d_ff d_model"], 
        w2_weight: Float[Tensor, " d_model d_ff"], 
        w3_weight: Float[Tensor, " d_ff d_model"]
    ):
        state_dict = {
            "w1.weight": w1_weight, 
            "w2.weight": w2_weight, 
            "w3.weight": w3_weight
        }
        self.load_state_dict(state_dict)

    def silu(self, x: Float[Tensor, " ... d_ff"]) -> Float[Tensor, " ... d_ff"]:
        return x * torch.sigmoid(x)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        up_res = self.w1.forward(x)
        silu_res = self.silu(up_res)
        gate_res = silu_res * self.w3.forward(x)
        down_res = self.w2.forward(gate_res)
        return down_res

class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None, dtype = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.construct_cache()

    def construct_cache(self):
        cache = torch.empty((self.max_seq_len, self.d_k // 2, 2, 2), **self.factory_kwargs)
        for i in range(self.max_seq_len):
            for k in range(self.d_k // 2):
                cache[i][k] = self.cal_block_mat(i, k)
        # rotation_matrix 是可重建缓存，不需要从 checkpoint 恢复。如果持久化，后续 self attention 加载权重时没有 rotation matrix 会报错。
        self.register_buffer("rotation_matrix", cache, persistent=False)

    def cal_angle(self, pos: int, k: int):
        return pos / self.theta ** ((2 * k) / self.d_k)

    def cal_block_mat(self, pos: int, k: int) -> Tensor:
        angle = self.cal_angle(pos, k)
        return torch.tensor(
            [
                cos(angle),
                -sin(angle),
                sin(angle),
                cos(angle),
            ],
            **self.factory_kwargs
        ).reshape((2, 2))

    def forward(
        self, 
        x: Float[Tensor, " ... seq_len d_k"], 
        token_pos: Int[Tensor, " ... seq_len"]
    ) -> Float[Tensor, " ... seq_len d_k"]:
        n1 = self.d_k // 2
        n2 = 2
        # ... seq_len d_k -> ... seq_len d_k/2 2
        reshape_x = rearrange(
            x, " ... seq_len (n1 n2)-> ... seq_len n1 n2", n1 = n1, n2 = n2
        )
        # self.rotation_matrix: seq_len d_k/2 2 2
        # rotation_matrix: d_k/2 2 2
        ro_matrix = self.rotation_matrix[token_pos]
        res = einsum(reshape_x, ro_matrix, " ... n1 j, ... n1 i j-> ... n1 i")
        return res.reshape(*x.shape)

def softmax(x: Float[Tensor, " ..."], dim: int, temperature: float = 1.0) -> Float[Tensor, " ..."]:
    # 由于 softmax 对“给所有输入同时加上同一个常数”是不变的，取指定维度的 max，用 x[i] - max 来保证 x[i].exp() ∈ (0, 1]，降低 exp 溢出 inf 的风险
    max = torch.max(x, dim = dim, keepdim=True).values
    shifted = torch.sub(x, max)
    exp = torch.exp(shifted / temperature)
    sum = torch.sum(exp, dim = dim, keepdim=True)
    return exp / sum

def scaled_dot_product_attention(
        Q: Float[Tensor, " ... q d_k"],
        K: Float[Tensor, " ... k d_k"],
        V: Float[Tensor, " ... k d_v"],
        mask: Bool[Tensor, " ... q k"] | None = None,
    ) -> Float[Tensor, " ... q d_v"]:
    d_k = Q.shape[-1]
    dot_product = einsum(Q, K, " ... q d_k, ... k d_k-> ... q k")
    # scale
    dot_product = dot_product / sqrt(d_k)

    if mask is not None:
        factory_kwargs = {"device": Q.device, "dtype": Q.dtype}
        dot_product = torch.where(
            mask,
            dot_product,
            # -inf becomes zero after softmax
            torch.tensor(float("-inf"), **factory_kwargs)
        )

    score = softmax(dot_product, dim = -1)
    return einsum(score, V, " ... q k, ... k d_v-> ... q d_v")

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.factory_kwargs = {"device": device, "dtype": dtype}

        if max_seq_len != 0:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device, dtype)

        self.q_proj = Linear(self.d_model, self.d_model, device, dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device, dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device, dtype)
        self.output_proj = Linear(self.d_model, self.d_model, device, dtype)

    def load_weight(
        self, 
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        output_proj_weight: Float[Tensor, " d_model d_v"],
    ):
        state_dict = {
            "q_proj.weight": q_proj_weight,
            "k_proj.weight": k_proj_weight,
            "v_proj.weight": v_proj_weight,
            "output_proj.weight": output_proj_weight,
        }
        # 需要（从 checkpoint）加载所有权重
        self.load_state_dict(state_dict)

    def concat_heads(self, x: Float[Tensor, "... head_dim seq_len d_k"]) -> Float[Tensor, "... seq_len d_model"]:
        # 重排维度 + 合并维度，可以用 torch.transpose + torch.flatten/reshape 或 einops.rearrange
        # 交换 head_dim、seq_len 维的位置（只换轴顺序，不改元素总和），把一段连续维度合并成一个维度（可能拷贝，不改元素总和）
        # return x.transpose(-2, -3).flatten(start_dim=-2)
        return rearrange(x, "... head_dim seq_len d_k -> ... seq_len (head_dim d_k)")

    def split_heads(self, x: Float[Tensor, "... seq_len d_in"]) -> Float[Tensor, "... head_dim seq_len d_k"]:
        # new_shape = x.shape[:-1] + (self.num_heads, self.d_k)
        # return x.view(new_shape).transpose(-2, -3)
        return rearrange(x, "... seq_len (head_dim d_k) -> ... head_dim seq_len d_k", head_dim = self.num_heads)
    
    def construct_causal_mask(self, seq_len: int) -> Bool[Tensor, "seq_len seq_len"]:
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), **self.factory_kwargs),
            diagonal=1,
        ).bool()
        # True in mask means visible, which can see itself and the past
        return ~causal_mask

    def forward(
        self, 
        x: Float[Tensor, " ... seq_len d_in"],
        token_pos: Int[Tensor, " ... seq_len"] | None = None,
    ) -> Float[Tensor, " ... seq_len d_out"]:
        q_proj = self.split_heads(self.q_proj.forward(x))
        k_proj = self.split_heads(self.k_proj.forward(x))
        v_proj = self.split_heads(self.v_proj.forward(x))

        # head_dim should be handled as a batched dimension, because in multi-head attention, attention is being applied independently for each head
        if token_pos is not None:
            # using unsqueeze to add head dim to the 1st dimension
            q_proj = self.rope.forward(q_proj, token_pos.unsqueeze(1))
            k_proj = self.rope.forward(k_proj, token_pos.unsqueeze(1))

        seq_len = x.shape[-2]
        causal_mask = self.construct_causal_mask(seq_len)
        head_attn = scaled_dot_product_attention(q_proj, k_proj, v_proj, causal_mask)
        attn = self.concat_heads(head_attn)
        return self.output_proj.forward(attn)

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device = None, dtype = None):
        super().__init__()
        eps = 1e-5
        self.ln1 = RMSNorm(d_model, eps, device, dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device, dtype)
        self.ln2 = RMSNorm(d_model, eps, device, dtype)
        self.ffn = SwiGLUFFN(d_model, d_ff, device, dtype)

        # 位置就是等差数列，预存以降低重复生成开销
        token_pos = torch.arange(max_seq_len, device = device, dtype = torch.long)
        self.register_buffer("token_pos", token_pos, persistent=False)

    def load_weight(self, weights: dict[str, Tensor]):
        self.load_state_dict(weights)
    
    def forward(self, x: Float[Tensor, "batch seq_len d_model"]) -> Float[Tensor, "batch seq_len d_model"]:
        batch, seq_len, _ = x.shape
        # 在预存的位置中，取前 s 个位置，形状为 (s,)。
        # 在最前面插入长度为 1 的维度，形状为 (1, s)，对齐位置索引和 batch 维并自动广播到每个样本。
        token_pos = self.token_pos[:seq_len].unsqueeze(0)

        y = x + self.attn.forward(self.ln1.forward(x), token_pos)
        y2 = y + self.ffn.forward(self.ln2.forward(y))
        return y2

class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device = None, dtype = None) -> None:
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype)
            )
        self.ln_final = RMSNorm(d_model, 1e-5, device, dtype)
        self.lm_head = Linear(d_model, vocab_size)

    def load_weight(self, weights: dict[str, Tensor]):
        self.load_state_dict(weights)

    def forward(self, in_indices: Int[Tensor, " batch seq_len"]) -> Float[Tensor, " batch seq_len vocab_size"]:
        x = self.token_embeddings.forward(in_indices)
        for layer in self.layers:
            x = layer.forward(x)
        x = self.ln_final.forward(x)
        x = self.lm_head.forward(x)
        return x
            
def cross_entropy(inputs: Float[Tensor, " batch vocab"], targets: Int[Tensor, " batch"]) -> Float[Tensor, ""]:
    max = torch.max(inputs, dim = -1, keepdim = True).values
    inputs = inputs - max

    target_logits = torch.gather(inputs, dim = 1, index = targets.unsqueeze(1))

    loss = torch.log(torch.sum(torch.exp(inputs), dim = -1)) - target_logits
    return torch.mean(loss)

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3, betas = (0.9, 0.999), weight_decay = 0.01, eps = 1e-8):
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                
                grad = p.grad.data
                
                lr_adjusted = lr * (sqrt(1 - beta2**t) / (1 - beta1**t))
                p.data = p.data - lr * weight_decay * p.data
                
                m = state.get("m", torch.zeros_like(grad))
                v = state.get("v", torch.zeros_like(grad))
                m = beta1 * m + (1-beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                state["m"] = m
                state["v"] = v
                
                p.data = p.data - lr_adjusted * m / (torch.sqrt(v) + eps)
                
                state["t"] = t + 1
        return loss

def lr_cosine_schedule(t: int, lr_max: float, lr_min: float, t_w: int, t_c: int) -> float:
    if t < t_w:
        return (t / t_w) * lr_max
    elif t_w <= t and t <= t_c:
        return lr_min + 0.5 * (1 + cos((t - t_w) / (t_c - t_w) * pi)) * (lr_max - lr_min)
    else:
        return lr_min

def gradient_clipping(params: Iterable[torch.nn.Parameter], max_lr_norm: float):
    eps = 1e-6

    grads = sum([torch.sum(p.grad**2) for p in params if p.grad is not None])
    norm = torch.sqrt(grads)

    if norm >= max_lr_norm:
        scale_factor = max_lr_norm / (norm + eps)
        for p in params:
            if p.grad is None:
                continue
            p.grad.mul_(scale_factor)

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device = None) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(dataset) > context_length + 1
    indices = np.random.choice(
        len(dataset) - context_length, size=batch_size, replace=True,
    )
    # [0, 1, ..., context_length]
    offsets = np.arange(context_length + 1).reshape(1, -1)
    idx = indices.reshape(batch_size, 1) + offsets

    window = dataset[idx]
    x = window[:,:-1]
    y = window[:,1:]

    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return (x, y)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iter: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": iter
    }
    torch.save(state_dict, out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    return state_dict["iter"]