import sys
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
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
        model = model.to(device)
        self.sampler = DDIMSampler(model)
        return

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
        model = self.sampler.model

        seed = seed_to_int(seed)
        prng = np.random.RandomState(seed)
        start_code = prng.randn(n_iter, 4, height//8, width//8)
        start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            with torch.autocast("cuda"):
                batch = self.make_batch_sd(inpaint_img, inpaint_mask, txt=prompt, device=device, num_samples=n_iter)
                c = model.cond_stage_model.encode(batch["txt"])
                c_cat = list()

                for ck in model.concat_keys:
                    cc = batch[ck].float()

                    if ck != model.masked_image_key:
                        bchw = [n_iter, 4, height//8, width//8]
                        cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                    else:
                        cc = model.get_first_stage_encoding(model.encode_first_stage(cc))

                    c_cat.append(cc)

                c_cat = torch.cat(c_cat, dim=1)

                # cond
                cond={"c_concat": [c_cat], "c_crossattn": [c]}

                # uncond cond
                uc_cross = model.get_unconditional_conditioning(n_iter, "")
                uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

                shape = [model.channels, height//8, width//8]

                samples_cfg, intermediates = self.sampler.sample(
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

                x_samples_ddim = model.decode_first_stage(samples_cfg)
                result = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                result = result.cpu().numpy().transpose(0,2,3,1)
                result = result*255

        x_samples = [Image.fromarray(img.astype(np.uint8)) for img in result]

        for x_sample in x_samples:
            image_dict = {"seed": seed, "image": x_sample}
            self.images.append(image_dict)

        return
