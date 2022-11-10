import os
import re
import sys
import numpy as np
import torch
from PIL import Image
import PIL.ImageOps
from omegaconf import OmegaConf
from einops import repeat
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from slugify import slugify

from nataili.util import logger
from nataili.util.cache import torch_gc
from nataili.util.get_next_sequence_number import get_next_sequence_number
from nataili.util.save_sample import save_sample
from nataili.util.seed_to_int import seed_to_int


class inpainting:
    def __init__(
        self,
        device,
        output_dir,
        save_extension="jpg",
        output_file_path=False,
        load_concepts=False,
        concepts_dir=None,
        verify_input=True,
        auto_cast=True,
        filter_nsfw=False,
    ):
        self.output_dir = output_dir
        self.output_file_path = output_file_path
        self.save_extension = save_extension
        self.load_concepts = load_concepts
        self.concepts_dir = concepts_dir
        self.verify_input = verify_input
        self.auto_cast = auto_cast
        self.device = device
        self.comments = []
        self.output_images = []
        self.info = ""
        self.stats = ""
        self.images = []
        self.filter_nsfw = filter_nsfw


    def initialize_model(
        self,
        config,
        ckpt
    ):
        config = OmegaConf.load(config)
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(device)
        return


    def resize_image(self, resize_mode, im, width, height):
        LANCZOS = PIL.Image.Resampling.LANCZOS if hasattr(PIL.Image, "Resampling") else PIL.Image.LANCZOS
        if resize_mode == "resize":
            res = im.resize((width, height), resample=LANCZOS)
        elif resize_mode == "crop":
            ratio = width / height
            src_ratio = im.width / im.height

            src_w = width if ratio > src_ratio else im.width * height // im.height
            src_h = height if ratio <= src_ratio else im.height * width // im.width

            resized = im.resize((src_w, src_h), resample=LANCZOS)
            res = PIL.Image.new("RGBA", (width, height))
            res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
        else:
            ratio = width / height
            src_ratio = im.width / im.height

            src_w = width if ratio < src_ratio else im.width * height // im.height
            src_h = height if ratio >= src_ratio else im.height * width // im.width

            resized = im.resize((src_w, src_h), resample=LANCZOS)
            res = PIL.Image.new("RGBA", (width, height))
            res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

            if ratio < src_ratio:
                fill_height = height // 2 - src_h // 2
                res.paste(
                    resized.resize((width, fill_height), box=(0, 0, width, 0)),
                    box=(0, 0),
                )
                res.paste(
                    resized.resize(
                        (width, fill_height),
                        box=(0, resized.height, width, resized.height),
                    ),
                    box=(0, fill_height + src_h),
                )
            elif ratio > src_ratio:
                fill_width = width // 2 - src_w // 2
                res.paste(
                    resized.resize((fill_width, height), box=(0, 0, 0, height)),
                    box=(0, 0),
                )
                res.paste(
                    resized.resize(
                        (fill_width, height),
                        box=(resized.width, 0, resized.width, height),
                    ),
                    box=(fill_width + src_w, 0),
                )

        return res


    def make_batch_sd(
        self,
        image,
        mask,
        txt,
        device,
        num_samples=1
    ):
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

    def generate(
        self,
        prompt: str,
        inpaint_img=None,
        inpaint_mask=None,
        ddim_steps=50,
        n_iter=1,
        batch_size=1,
        cfg_scale=7.5,
        seed=None,
        height=512,
        width=512,
        save_individual_images: bool = True,
    ):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        sampler = DDIMSampler(self.model)

        inpaint_img = self.resize_image("resize", inpaint_img, width, height)

        # mask information has been transferred in the Alpha channel of the inpaint image
        logger.debug(inpaint_mask)
        if inpaint_mask is None:
            try:
                red, green, blue, alpha = inpaint_img.split()
            except ValueError:
                raise Exception("inpainting image doesn't have an alpha channel.")

            inpaint_mask = alpha
            inpaint_mask = PIL.ImageOps.invert(inpaint_mask)
        else:
            inpaint_mask = self.resize_image("resize", inpaint_mask, width, height)

        seed = seed_to_int(seed)
        prng = np.random.RandomState(seed)
        start_code = prng.randn(n_iter, 4, height//8, width//8)
        start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

        if self.load_concepts and self.concepts_dir is not None:
            prompt_tokens = re.findall("<([a-zA-Z0-9-]+)>", prompt)
            if prompt_tokens:
                self.process_prompt_tokens(prompt_tokens)

        os.makedirs(self.output_dir, exist_ok=True)

        sample_path = os.path.join(self.output_dir, "samples")
        os.makedirs(sample_path, exist_ok=True)

        all_prompts = batch_size * [prompt]
        all_seeds = [seed + x for x in range(len(all_prompts))]

        torch_gc()

        with torch.no_grad(), torch.autocast("cuda"):
            for n in range(batch_size):
                print(f"Iteration: {n+1}/{batch_size}")

                prompt = all_prompts[n]
                seed = all_seeds[n]
                print("prompt: " + prompt + ", seed: " + str(seed))

                batch = self.make_batch_sd(inpaint_img, inpaint_mask, txt=prompt, device=device, num_samples=n_iter)
                c = self.model.cond_stage_model.encode(batch["txt"])
                c_cat = list()

                for ck in self.model.concat_keys:
                    cc = batch[ck].float()

                    if ck != self.model.masked_image_key:
                        bchw = [n_iter, 4, height//8, width//8]
                        cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                    else:
                        cc = self.model.get_first_stage_encoding(self.model.encode_first_stage(cc))

                    c_cat.append(cc)

                c_cat = torch.cat(c_cat, dim=1)

                # cond
                cond={"c_concat": [c_cat], "c_crossattn": [c]}

                # uncond cond
                uc_cross = self.model.get_unconditional_conditioning(n_iter, "")
                uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

                shape = [self.model.channels, height//8, width//8]

                samples_cfg, intermediates = sampler.sample(
                    ddim_steps,
                    n_iter,
                    shape,
                    cond,
                    verbose=False,
                    eta=1.0,
                    unconditional_guidance_scale=cfg_scale,
                    unconditional_conditioning=uc_full,
                    x_T=start_code,
                )

                x_samples_ddim = self.model.decode_first_stage(samples_cfg)
                result = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                result = result.cpu().numpy().transpose(0,2,3,1)
                result = result*255

                x_samples = [Image.fromarray(img.astype(np.uint8)) for img in result]

                for i, x_sample in enumerate(x_samples):
                    image_dict = {"seed": seed, "image": x_sample}
                    self.images.append(image_dict)
                    
                    if save_individual_images:
                        sanitized_prompt = slugify(prompt)
                        sample_path_i = sample_path
                        base_count = get_next_sequence_number(sample_path_i)
                        full_path = os.path.join(os.getcwd(), sample_path)
                        filename = f"{base_count:05}-{ddim_steps}_{seed}_{sanitized_prompt}"[: 200 - len(full_path)]

                        path = os.path.join(sample_path, filename + "." + self.save_extension)
                        success = save_sample(x_sample, filename, sample_path_i, self.save_extension)

                        if success:
                            if self.output_file_path:
                                self.output_images.append(path)
                            else:
                                self.output_images.append(x_sample)
                        else:
                            return

        self.info = f"""
                {prompt}
                Steps: {ddim_steps}, CFG scale: {cfg_scale}, Seed: {seed}
                """.strip()
        self.stats = """
                """

        for comment in self.comments:
            self.info += "\n\n" + comment

        torch_gc()

        return
