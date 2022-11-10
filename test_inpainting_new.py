from PIL import Image
from nataili.inference.inpainting import inpainting
from nataili import disable_xformers

config = "configs/stable-diffusion/v1-inpainting-inference.yaml"
ckpt = "models/sd-v1-5-inpainting.ckpt"
original = Image.open("./inpaint_original.png")
mask = Image.open("./inpaint_mask.png")
prompt = "a robot sitting on a bench"

disable_xformers.toggle(True)
generator = inpainting("cuda", "output_dir")
generator.initialize_model(config, ckpt)
generator.generate(prompt, original, mask)
image = generator.images[0]["image"]
image.save("robot_sitting_on_a_bench.png", format="PNG")
