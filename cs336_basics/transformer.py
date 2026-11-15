from typing import Any

from einops import einsum, rearrange
import torch
from torch import Tensor
from math import cos, sin, sqrt
from jaxtyping import Float, Int

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

    def __init__(self, d_model: int, eps: float, device = None, dtype = None) -> None:
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
        self.register_buffer("rotation_matrix", cache, persistent=True)

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
        rotation_matrix = self.rotation_matrix[token_pos]
        res = einsum(reshape_x, rotation_matrix, " ... n1 j, ... n1 i j-> ... n1 i")
        return res.reshape(*x.shape)
