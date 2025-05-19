# Copyright 2022 Synnada, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This example is equivalent of nanoGPT by karpathy.
Torch implementation of nanoGPT: https://github.com/karpathy/nanoGPT
"""

from mithril import IOKey
from mithril.models import (
    Arange,
    Embedding,
    Gelu,
    LayerNorm,
    Linear,
    Model,
    ScaledDotProduct,
    Size,
    Split,
    functional,
)


@functional
def attn(input: IOKey, *, input_dim: int, num_heads: int, bias: bool = True):
    if (input_dim % num_heads) != 0:
        raise ValueError("Requires input dims to be divisible by num_heads")

    lin_out = Linear(input_dim * 3, name="c_attn")(input=input)

    t_axes = (0, 2, 1, 3)
    shp_con = input.shape
    reshape_con = (shp_con[0], shp_con[1], num_heads, -1)

    split_out = Split(3, axis=-1)(input=lin_out)
    tq = split_out[0].reshape(reshape_con).transpose(t_axes)
    tk = split_out[1].reshape(reshape_con).transpose(t_axes)
    tv = split_out[2].reshape(reshape_con).transpose(t_axes)

    sdp_out = ScaledDotProduct()(query=tq, key=tk, value=tv)
    t_sdp = sdp_out.transpose(t_axes).reshape(shp_con[:3])
    output = Linear(input_dim, name="c_proj")(input=t_sdp)
    return output


@functional
def mlp(input: IOKey, *, n_embed: int):
    fc_out = Linear(n_embed * 4, name="c_fc")(input=input)
    gelu_out = Gelu()(input=fc_out)
    output = Linear(n_embed, name="c_proj")(input=gelu_out)
    return output


@functional
def create_block(
    input: IOKey, *, dims: int, num_heads: int, bias: bool = True, eps: float = 1e-5
):
    input = IOKey("input")
    ln1_out = LayerNorm(use_bias=bias, eps=eps, name="ln_1")(input=input)
    attn_out = attn(input=ln1_out, input_dim=dims, num_heads=num_heads, bias=bias)
    add1_out = input + attn_out
    ln2_out = LayerNorm(use_bias=bias, eps=eps, name="ln_2")(input=add1_out)
    mlp_out = mlp(ln2_out, n_embed=dims)
    output = add1_out + mlp_out
    return output


@functional
def h(input: IOKey, *, dims: int, num_heads: int, num_layers: int):
    output = input
    for idx in range(num_layers):
        output = create_block(output, name=f"{idx}", dims=dims, num_heads=num_heads)

    return output


@functional
def transformer(
    input: IOKey,
    *,
    bias: bool,
    block_size: int,
    dims: int,
    num_heads: int,
    num_layers: int,
    vocab_size: int,
):
    s_out = Size(dim=1)(input=input)
    arr_out = Arange(start=0, step=1)(stop=s_out)
    pos_out = Embedding(name="wpe", num_embeddings=block_size, dim=dims)(input=arr_out)
    token_out = Embedding(name="wte", num_embeddings=vocab_size, dim=dims)(input=input)

    blocks_out = h(
        pos_out + token_out, dims=dims, num_heads=num_heads, num_layers=num_layers
    )
    ln_out = LayerNorm(use_bias=bias, name="ln_f")(input=blocks_out)
    return ln_out


def create_gpt(
    bias: bool,
    block_size: int,
    dims: int,
    num_heads: int,
    num_layers: int,
    vocab_size: int,
) -> Model:
    input = IOKey("input", differentiable=False)
    t_out = transformer(
        input=input,
        block_size=block_size,
        dims=dims,
        num_heads=num_heads,
        num_layers=num_layers,
        bias=bias,
        vocab_size=vocab_size,
    )
    output = Linear(vocab_size, use_bias=False, name="lm_head")(input=t_out)
    return Model.create(output=output)
