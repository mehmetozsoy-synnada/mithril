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
from diffusers import FluxFillPipeline
FluxFillPipeline.from_pretrained

from t5 import download_t5_encoder_weights, load_t5_encoder, load_t5_tokenizer
from util import load_decoder, load_encoder, load_flow_model

import mithril as ml
from mithril.models import (
    IOKey,
)
import pickle

from PIL import Image
import numpy as np

def main(
    seed: int = 42,
    prompt: str = "a white paper cup",
    device: str = "cuda",
    num_steps: int = 30,
    guidance: float = 30.0,
    output_dir: str = "output",
    img_cond_path: str = "examples/flux/assets/cup.png",
    img_mask_path: str = "examples/flux/assets/cup_mask.png",
):
        
    backend = ml.TorchBackend(device=device, dtype=ml.bfloat16)
    img_cond = Image.open(img_cond_path)
    img_cond = img_cond.crop((0, 0, 1024, 1024))


    img_mask = Image.open(img_mask_path)
    img_mask = img_cond.crop((0, 0, 1024, 1024))
        
    img_cond = np.array(img_cond.convert("RGB"))
    img_mask = np.array(img_mask.convert("L"))

    height = img_cond.shape[0]
    width = img_cond.shape[1]


    name = "flux-dev-fill"

    print("Loading T5 encoder")
    t5_lm = load_t5_encoder(name, 128)
    t5_tokenizer = load_t5_tokenizer(backend, name)  # noqa F841
    t5_weights = download_t5_encoder_weights(backend, name)  # noqa F841
    t5_lm.name = "t5"
    
    print("Loading CLIP encoder")
    clip_lm = load_clip_encoder(name)
    clip_tokenizer = load_clip_tokenizer(backend, name)  # noqa F841
    clip_weights = download_clip_encoder_weights(backend, name)  # noqa F841
    clip_lm.name = "clip"
    clip_lm.set_cout("output")

    decoder_lm, decoder_params = load_decoder(name, backend, width, height)  # noqa F841
    decoder_lm.name = "decoder"

    encoder_lm, encoder_params = load_encoder(name,backend, width, height)  # noqa F841
    encoder_lm.name = "encoder"

    flow_lm, flow_params = load_flow_model(name, backend=backend)  # noqa F841
    
    clip_inp = clip_tokenizer.encode(prompt)
    t5_inp = t5_tokenizer.encode(prompt)
    
    model_inputs = {
        "t5_tokens": t5_inp,
        "clip_tokens": clip_inp,
        "img_cond": backend.array(img_cond, dtype = ml.bfloat16),
        "mask":  backend.array(img_mask, dtype = ml.bfloat16)
    }
    
    flow_params = {f"model_0_{key}": value for key, value in flow_params.items()}
    decoder_params = {f"decoder_{key}": value for key, value in decoder_params.items()}
    encoder_params = {f"encoder_{key}": value for key, value in encoder_params.items()}
    t5_params = {f"t5_{key}": value for key, value in t5_weights.items()}
    clip_params = {f"clip_{key}": value for key, value in clip_weights.items()}
    
    all_params = {**flow_params, **decoder_params, **encoder_params, **t5_params, **clip_params}
    
    print("get_noise")
    input = get_noise(
        1,
        height,
        width,
        seed,
    )

    print("prep fill")
    kwargs = prepare_fill(t5=t5_lm, clip=clip_lm, img=input, encoder=encoder_lm)
    

    print("get_schedule")
    timesteps = get_schedule(
        num_steps,
        kwargs["img"].metadata.shape.get_shapes()[1],  # type: ignore
        shift=True,
        backend=backend,
    )
    denoise_output = denoise_logical(
        flow_lm,
        timesteps=timesteps,
        guidance=guidance,
        ** kwargs
    )
    print("unpack_output")
    unpacked_output = unpack(denoise_output, height, width)
    print("decode_output")
    
    print(f"unpacked output: {unpacked_output.metadata.shape.get_shapes()}")
    print(f"decocde model input: {decoder_lm.cin.metadata.shape.get_shapes()}")
    decoded_output = decoder_lm(input = unpacked_output)

    print("create_model")
    finalized_model = ml.models.Model.create(output=decoded_output.transpose((0, 2, 3, 1)))
    
    print("compile")
    denoise_pm = ml.compile(
        finalized_model, backend, inference=True, jit=False, use_short_namings=False
    )
    denoise_pm.summary()
    print("evaluate")
    output, state = denoise_pm.evaluate(all_params, model_inputs, state = denoise_pm.initial_state_dict)
    img_pil = Image.fromarray(
        np.array(127.5 * (output["output"].float().cpu()[0] + 1.0)).clip(0, 255).astype(np.uint8)  # type: ignore
    )
    img_pil.save("img.png")
    
    


main()
