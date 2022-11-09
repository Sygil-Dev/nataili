import sys
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
import torch
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

MAX_SIZE = 640

def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    return sampler

  
def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(dtype=torch.float32)/127.5-1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
            "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
            "txt": num_samples * [txt],
            "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
            "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
            }
    return batch


def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h//8, w//8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        with torch.autocast("cuda"):
            batch = make_batch_sd(image, mask, txt=prompt, device=device, num_samples=num_samples)

            c = model.cond_stage_model.encode(batch["txt"])

            c_cat = list()
            for ck in model.concat_keys:
                cc = batch[ck].float()
                if ck != model.masked_image_key:
                    bchw = [num_samples, 4, h//8, w//8]
                    cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                else:
                    cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)

            # cond
            cond={"c_concat": [c_cat], "c_crossattn": [c]}

            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

            shape = [model.channels, h//8, w//8]
            samples_cfg, intermediates = sampler.sample(
                    ddim_steps,
                    num_samples,
                    shape,
                    cond,
                    verbose=False,
                    eta=1.0,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc_full,
                    x_T=start_code,
            )
            x_samples_ddim = model.decode_first_stage(samples_cfg)

            result = torch.clamp((x_samples_ddim+1.0)/2.0,
                                 min=0.0, max=1.0)

            result = result.cpu().numpy().transpose(0,2,3,1)
            result = result*255

    result = [Image.fromarray(img.astype(np.uint8)) for img in result]
    return result


def run(config, ckpt, image, mask, prompt):
    print("Start inpainting...")
    sampler = initialize_model(config, ckpt)

    seed = 123456
    num_samples = 1
    scale = 7.5
    ddim_steps = 50
    width = 512
    height = 512

    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")

    if max(w, h) > MAX_SIZE:
       factor = MAX_SIZE / max(w, h)
       w = int(factor*w)
       h = int(factor*h)
       width, height = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
       image = image.resize((width, height))
       print(f"resized to ({width}, {height})")

    result = inpaint(
       sampler=sampler,
       image=image,
       mask=mask,
       prompt=prompt,
       seed=seed,
       scale=scale,
       ddim_steps=ddim_steps,
       num_samples=num_samples,
       h=height, w=width
    )
 
    return result
