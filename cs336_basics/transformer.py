from typing import Any

from einops import einsum
import torch
from torch import Tensor
from math import sqrt
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
