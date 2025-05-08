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

import math
from collections.abc import Callable
from copy import deepcopy

import torch

import mithril as ml
from mithril.models import (
    Arange,
    BroadcastTo,
    Concat,
    IOKey,
    Multiply,
    Ones,
    Randn,
    Reshape,
)


def prepare_logical(
    block: ml.models.Model,
    t5: ml.models.Model,
    clip: ml.models.Model,
    num_samples: int,
    height: int,
    width: int,
):
    c = 16
    h = 2 * math.ceil(height / 16)
    w = 2 * math.ceil(width / 16)

    block |= Randn(shape=(num_samples, (h // 2) * (w // 2), c * 2 * 2)).connect(
        output=IOKey("img")
    )

    block |= Ones(shape=(num_samples, h // 2, w // 2, 1)).connect(output="ones")
    block |= Multiply().connect(left="ones", right=0, output="img_ids_preb")
    block |= Arange(stop=(w // 2)).connect(output="arange_1")
    block |= BroadcastTo(shape=(num_samples, h // 2, w // 2)).connect(
        block.arange_1[None, :, None],  # type: ignore
        output="arange_1_bcast",
    )
    block |= Arange(stop=(h // 2)).connect(output="arange_2")
    block |= BroadcastTo(shape=(num_samples, h // 2, w // 2)).connect(
        block.arange_2[None, None, :],  # type: ignore
        output="arange_2_bcast",
    )
    block |= Concat(axis=-1).connect(
        input=[
            block.img_ids_preb,  # type: ignore
            block.arange_1_bcast[..., None],  # type: ignore
            block.arange_2_bcast[..., None],  # type: ignore
        ],
        output="img_ids_cat",
    )

    block |= Reshape(shape=(num_samples, -1, 3)).connect(
        block.img_ids_cat,  # type: ignore
        output=IOKey("img_ids"),
    )

    block |= t5.connect(input=IOKey("t5_tokens"), output=IOKey("txt"))
    block |= Ones().connect(
        shape=(num_samples, block.txt.shape[1], 3),  # type: ignore
        output="txt_ids_preb",
    )
    block |= Multiply().connect(left="txt_ids_preb", right=0, output=IOKey("txt_ids"))

    block |= clip.connect(input=IOKey("clip_tokens"), output=IOKey("y"))


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
    *,
    backend: ml.Backend,
) -> list[float]:
    # extra step for zero
    timesteps = backend.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def unpack(
    input: ml.models.Connection, height: int, width: int
) -> ml.models.Connection:
    h = math.ceil(height / 16)
    w = math.ceil(width / 16)
    b = input.shape[0]

    input = input.reshape((b, h, w, -1, 2, 2))
    input = input.transpose((0, 3, 1, 4, 2, 5))
    input = input.reshape((b, -1, 2 * h, 2 * w))

    return input


def denoise(
    model: ml.models.PhysicalModel,
    params: dict,
    # model input
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    vec: torch.Tensor,
    # sampling parameters
    timesteps: list[float],
    backend: ml.Backend,
    guidance: float = 4.0,
):
    # this is ignored for schnell
    guidance_vec = backend.ones((img.shape[0],)) * guidance
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:], strict=False):
        t_vec = backend.ones((img.shape[0],)) * t_curr
        pred = model.evaluate(
            params,
            {
                "img": img,
                "img_ids": img_ids,
                "txt": txt,
                "txt_ids": txt_ids,
                "y": vec,
                "timesteps": t_vec,
                "guidance": guidance_vec,
            },
        )

        img = img + (t_prev - t_curr) * pred["output"]  # type: ignore[operator]

    return img


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    seed: int,
) -> ml.models.Connection:
    height = 2 * math.ceil(height / 16)
    width = 2 * math.ceil(width / 16)
    noise_model = Randn(shape=(num_samples, 16, height, width))
    return noise_model.output


def prepare(
    t5: ml.models.Model, clip: ml.models.Model, img: ml.models.Connection
) -> dict[str, ml.models.Connection]:
    bs = img.shape[0]
    c = img.shape[1]
    h = img.shape[2]
    w = img.shape[3]

    img = img.reshape((bs, c, h // 2, 2, w // 2, 2))
    img = img.transpose((0, 2, 4, 1, 3, 5))
    img = img.reshape((bs, -1, c * 4))

    img_id_1 = Ones()(shape=(h // 2, w // 2)) * 0.0
    
    img_id_2 = Arange()(stop=(h // 2))[:, None]
    img_id_2 = BroadcastTo()(input = img_id_2, shape = (h // 2, w // 2))
    
    img_id_3 = Arange()(stop=(w // 2))[None, :]
    img_id_3 = BroadcastTo()(input = img_id_2, shape = (h // 2, w // 2))
    
    img_ids = Concat(axis=-1)(input = [img_id_1[..., None], img_id_2[..., None], img_id_3[..., None]])
    
    img_ids = img_ids.reshape((bs, -1, 3))

    txt = t5(input = IOKey("t5_tokens"))
    txt_ids = Ones()(shape=(bs, txt.shape[1], 3)) * 0.0

    vec = clip(input = IOKey("clip_tokens"))
    
    return {
        "img": img,
        "img_ids": img_ids, 
        "txt": txt, 
        "txt_ids": txt_ids,
        "vec": vec
    }


def prepare_fill(
    t5: ml.models.Model,
    clip: ml.models.Model,
    img: ml.models.Connection,
    encoder: ml.models.Model,
):
    img_cond = IOKey("img_cond")
    img_cond = img_cond / 127.5 - 1.0
    img_cond = img_cond.transpose((2, 0, 1))[None, ...]

    mask = IOKey("mask")
    mask = mask / 255.0
    mask = mask[None, None, ...]
    img_cond = img_cond * (1 - mask)
    img_cond = encoder(input=img_cond)

    mask = mask[:, 0, :, :]
    b_mask = mask.shape[0]
    h_mask = mask.shape[1]
    w_mask = mask.shape[2]

    # rearrange(mask,"b (h ph) (w pw) -> b (ph pw) h w", ph=8, pw=8)
    mask = mask.reshape((b_mask, h_mask // 8, 8, w_mask // 8, 8))
    mask = mask.transpose((0, 2, 4, 1, 3))
    mask = mask.reshape((b_mask, 64, h_mask // 8, w_mask // 8))

    # rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    mask = mask.reshape((b_mask, 64, h_mask // 16, 2, w_mask // 16, 2))
    mask = mask.transpose((0, 2, 4, 1, 3, 5))
    mask = mask.reshape((b_mask, -1, 256))

    b_cond = img_cond.shape[0]
    c_cond = img_cond.shape[1]
    h_cond = img_cond.shape[2]
    w_cond = img_cond.shape[3]

    # rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    img_cond = img_cond.reshape((b_cond, c_cond, h_cond // 2, 2, w_cond // 2, 2))
    img_cond = img_cond.transpose((0, 2, 4, 1, 3, 5))
    img_cond = img_cond.reshape((b_cond, -1, c_cond * 4))

    img_cond = Concat(axis=-1)(input = [img_cond, mask])

    kwargs = prepare(t5, clip, img)
    kwargs["img_cond"] = img_cond
    return kwargs


def denoise_logical(
    flux_model: ml.models.Model,
    img: ml.models.Connection,
    img_ids: ml.models.Connection,
    txt: ml.models.Connection,
    txt_ids: ml.models.Connection,
    vec: ml.models.Connection,
    timesteps: list[float],
    guidance: float = 4.0,
    img_cond: ml.models.Connection | None = None,
):
    guidance_vec = Ones()(shape=[img.shape[0]]) * guidance

    weigth_kwargs = {
        key: IOKey()
        for key in flux_model.input_keys
        if "$" in key
    }
    i = 0
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:], strict=False):
        print(f"denoise step {i}")
        t_vec = Ones()(shape=[img.shape[0]]) * t_curr
        _flux_model = deepcopy(flux_model)
        

        pred = _flux_model(
            img=Concat(axis = -1)((img, img_cond)) if img_cond is not None else img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            **weigth_kwargs,
        )
        print(f"img shape: {img.metadata.shape.get_shapes()}")
        print(f"pred shape: {pred.metadata.shape.get_shapes()}")
        img = img + (t_prev - t_curr) * pred
        i += 1

    return pred
