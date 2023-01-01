import uuid

import PIL

from nataili import CompVis, ModelManager, logger

init_image = PIL.Image.open("./01.png").convert("RGB")

mm = ModelManager()

models_to_load = [
    "stable_diffusion",
]
logger.init(f"{models_to_load}", status="Loading")


def test_compvis(
    model,
    prompt,
    sampler,
    steps=30,
    output_dir=None,
    seed=None,
    init_img=None,
    denoising_strength=0.75,
    sigma_override=None,
    filter_nsfw=False,
    safety_checker=None,
):
    log_message = f"sampler: {sampler} steps: {steps} model: {model}"
    if seed:
        log_message += f" seed: {seed}"
    if sigma_override:
        log_message += f" sigma_override: {sigma_override}"
    logger.info(log_message)
    if filter_nsfw:
        logger.info("Filtering NSFW")
    if init_img:
        logger.info("Using init image")
        logger.info(f"Denoising strength: {denoising_strength}")
    compvis = CompVis(
        model=mm.compvis.loaded_models[model],
        model_name=model,
        output_dir=output_dir,
        disable_voodoo=True,
        filter_nsfw=filter_nsfw,
        safety_checker=safety_checker,
    )
    compvis.generate(
        prompt,
        sampler_name=sampler,
        ddim_steps=steps,
        seed=seed,
        init_img=init_img,
        sigma_override=sigma_override,
        denoising_strength=denoising_strength,
    )


samplers = [
    "k_dpm_fast",
    "k_dpmpp_2s_a",
    "k_dpm_adaptive",
    "k_dpmpp_2m",
    "k_dpm_2_a",
    "k_dpm_2",
    "k_euler_a",
    "k_euler",
    "k_heun",
    "k_lms",
]

step_counts = [5, 7, 8, 15]

sigma_overrides = [
    {"min": 0.6958, "max": 9.9172, "rho": 7.0},
]

denoising_strengths = [0.22, 0.44, 0.88]

output_dir = f"./test_output/{str(uuid.uuid4())}"


@logger.catch
def test():
    for model in models_to_load:
        mm.compvis.load(model)

        logger.info(f"Output dir: {output_dir}")
        logger.debug(f"Running inference on {model}")
        logger.info(f"Testing {len(samplers)} samplers")
        prompt = (
            "Headshot of cybernetic female character, cybernetic implants, solid background color,"
            "digital art, illustration, smooth color, cinematic moody lighting, cyberpunk, body modification,"
            "wenjun lin, studio ghibli, pixiv, artgerm, greg rutkowski, ilya kuvshinov"
        )
        logger.info(f"Prompt: {prompt}")
        for sampler in samplers:
            test_compvis(model, prompt, sampler, output_dir=output_dir)

        samplers.remove("k_dpm_adaptive")  # This sampler doesn't work with step counts
        logger.info(f"Testing {len(samplers)} samplers with Karras on step counts {step_counts}")
        logger.info(f"Prompt: {prompt}")
        for sampler in samplers:
            sampler = f"{sampler}_karras"
            for steps in step_counts:
                test_compvis(model, prompt, sampler, steps=steps, output_dir=output_dir)

        logger.info(f"Testing {len(samplers)} samplers with Karras on step counts {step_counts} and sigma overrides")
        logger.info(f"Prompt: {prompt}")
        for sampler in samplers:
            sampler = f"{sampler}_karras"
            for steps in step_counts:
                for sigma_override in sigma_overrides:
                    test_compvis(
                        model, prompt, sampler, steps=steps, sigma_override=sigma_override, output_dir=output_dir
                    )

        prompt = "cute anime girl"
        logger.info(f"Prompt: {prompt}")
        for denoising_strength in denoising_strengths:
            test_compvis(
                model,
                prompt,
                "k_lms",
                init_img=init_image,
                denoising_strength=denoising_strength,
                output_dir=output_dir,
            )


if __name__ == "__main__":
    test()
