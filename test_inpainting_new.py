from PIL import Image

from nataili.inference.inpainting import run

original = Image.open("./inpaint_original.png")
mask = Image.open("./inpaint_mask.png")

config = "configs/stable-diffusion/v1-inpainting-inference.yaml"
ckpt = "models/sd-v1-5-inpainting.ckpt"
images = run(config, ckpt, original, mask, prompt)
  
images[0].save("robot_sitting_on_a_bench.png", format="PNG")
