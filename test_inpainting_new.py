from PIL import Image

from nataili.inference.inpainting import run

config = "configs/stable-diffusion/v1-inpainting-inference.yaml"
ckpt = "models/sd-v1-5-inpainting.ckpt"
original = Image.open("./inpaint_original.png")
mask = Image.open("./inpaint_mask.png")
prompt = "a robot sitting on a bench"

images = run(config, ckpt, original, mask, prompt)

images[0].save("robot_sitting_on_a_bench.png", format="PNG")
