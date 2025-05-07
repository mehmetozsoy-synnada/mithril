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

from clip import download_clip_encoder_weights, load_clip_encoder, load_clip_tokenizer
from sampling import (
    denoise_logical,
    get_noise,
    get_schedule,
    prepare_fill,
    unpack,
)
from t5 import download_t5_encoder_weights, load_t5_encoder, load_t5_tokenizer
from util import load_decoder, load_encoder, load_flow_model

import mithril as ml
from mithril.models import (
    IOKey,
)


def main(
    seed: int = 37,
    prompt: str = "a white paper cup",
    device: str = "cpu",
    num_steps: int = 50,
    guidance: float = 30.0,
    output_dir: str = "output",
    img_cond_path: str = "assets/cup.png",
    img_mask_path: str = "assets/cup_mask.png",
):
    model_inputs = {
        "t5_tokens": IOKey("t5_tokens"),
        "clip_tokens": IOKey("clip_tokens"),
        "img_cond": IOKey("img_cond"),
        "mask": IOKey("mask"),
    }

    # with Image.open(img_cond_path) as img:
    #     width, height = img.size

    width = height = 1024

    backend = ml.TorchBackend(device=device, dtype=ml.bfloat16)

    name = "flux-dev-fill"

    print("Loading T5 encoder")
    t5_lm = load_t5_encoder(name, 512)
    t5_tokenizer = load_t5_tokenizer(backend, name)  # noqa F841
    t5_weights = download_t5_encoder_weights(backend, name)  # noqa F841
    t5_lm.name = "t5"

    print("Loading CLIP encoder")
    clip_lm = load_clip_encoder(name)
    clip_tokenizer = load_clip_tokenizer(backend, name)  # noqa F841
    clip_weights = download_clip_encoder_weights(backend, name)  # noqa F841
    clip_lm.name = "clip"

    decoder_lm, decoder_params = load_decoder(name, backend=backend)  # noqa F841
    decoder_lm.name = "decoder"

    encoder_lm, encoder_params = load_encoder(name, backend=backend)  # noqa F841
    encoder_lm.name = "encoder"

    flow_lm, flow_params = load_flow_model(name, backend=backend)  # noqa F841

    input = get_noise(
        1,
        height,
        width,
        seed,
    )

    prepare_model = prepare_fill(t5=t5_lm, clip=clip_lm, img=input, encoder=encoder_lm)
    img, img_ids, txt, txt_ids, vec, img_cond = prepare_model(**model_inputs)

    timesteps = get_schedule(
        num_steps,
        prepare_model.get_shapes()["img"][1],  # type: ignore
        shift=True,
        backend=backend,
    )
    denoise_output = denoise_logical(
        flow_lm,
        img=img,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        vec=vec,
        img_cond=img_cond,
        timesteps=timesteps,
        guidance=guidance,
    )(**model_inputs)
    unpacked_output = unpack(denoise_output, height, width)
    decoded_output = decoder_lm(unpacked_output)

    finalized_model = ml.models.Model.create(output=decoded_output)
    finalized_model.name = "fill"


main()
