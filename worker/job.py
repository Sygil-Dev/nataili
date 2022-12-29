"""Get and process a job from the horde"""
import base64
import copy
import json
import sys
import threading
import time
import traceback
from base64 import binascii
from io import BytesIO

import requests
from PIL import Image, UnidentifiedImageError

from nataili.inference.compvis import CompVis
from nataili.inference.diffusers.inpainting import inpainting
from nataili.util import logger
from worker.enums import JobStatus
from worker.post_process import post_process
from worker.stats import bridge_stats


class HordeJob:
    """Get and process a job from the horde"""

    retry_interval = 1

    def __init__(self, mm, bd):
        self.model_manager = mm
        self.bridge_data = copy.deepcopy(bd)
        self.current_id = None
        self.current_payload = None
        self.current_model = None
        self.loop_retry = 0
        self.status = JobStatus.INIT
        self.skipped_info = None
        self.upload_quality = 95
        self.start_time = time.time()
        self.seed = None
        self.image = None
        self.r2_upload = None
        self.pop = None
        self.submit_dict = {}
        self.headers = {"apikey": self.bridge_data.api_key}
        self.available_models = self.model_manager.get_loaded_models_names()
        for util_model in ["LDSR", "safety_checker", "GFPGAN", "RealESRGAN_x4plus", "CodeFormers"]:
            if util_model in self.available_models:
                self.available_models.remove(util_model)
        self.gen_dict = {
            "name": self.bridge_data.worker_name,
            "max_pixels": self.bridge_data.max_pixels,
            "priority_usernames": self.bridge_data.priority_usernames,
            "nsfw": self.bridge_data.nsfw,
            "blacklist": self.bridge_data.blacklist,
            "models": self.available_models,
            "allow_img2img": self.bridge_data.allow_img2img,
            "allow_painting": self.bridge_data.allow_painting,
            "allow_unsafe_ip": self.bridge_data.allow_unsafe_ip,
            "threads": self.bridge_data.max_threads,
            "bridge_version": 9,
        }

    def is_finished(self):
        """Check if the job is finished"""
        return self.status not in [JobStatus.WORKING, JobStatus.POLLING, JobStatus.INIT]

    def is_polling(self):
        """Check if the job is polling"""
        return self.status in [JobStatus.POLLING]

    def is_finalizing(self):
        """True if generation has finished even if upload is still remaining"""
        return self.status in [JobStatus.FINALIZING]

    def is_stale(self):
        """Check if the job is stale"""
        return time.time() - self.start_time > 1200

    def get_job_from_server(self):
        """Get a job from the horde"""
        self.status = JobStatus.POLLING
        try:
            pop_req = requests.post(
                self.bridge_data.horde_url + "/api/v2/generate/pop",
                json=self.gen_dict,
                headers=self.headers,
                timeout=20,
            )
            logger.debug(f"Job pop took {pop_req.elapsed.total_seconds()}")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Server {self.bridge_data.horde_url} unavailable during pop. Waiting 10 seconds...")
            time.sleep(10)
            self.status = JobStatus.FAULTED
        except TypeError:
            logger.warning(f"Server {self.bridge_data.horde_url} unavailable during pop. Waiting 2 seconds...")
            time.sleep(2)
            self.status = JobStatus.FAULTED
        except requests.exceptions.ReadTimeout:
            logger.warning(f"Server {self.bridge_data.horde_url} timed out during pop. Waiting 2 seconds...")
            time.sleep(2)
            self.status = JobStatus.FAULTED

        if self.status == JobStatus.FAULTED:
            return None
        try:
            pop = pop_req.json()
            self.pop = pop  # I'll use it properly later
        except json.decoder.JSONDecodeError:
            logger.error(
                f"Could not decode response from {self.bridge_data.horde_url} as json. "
                "Please inform its administrator!"
            )
            time.sleep(self.retry_interval)
            self.status = JobStatus.FAULTED
            return None
        if not pop_req.ok:
            logger.warning(
                f"During gen pop, server {self.bridge_data.horde_url} "
                f"responded with status code {pop_req.status_code}: "
                f"{pop['message']}. Waiting for 10 seconds..."
            )
            if "errors" in pop:
                logger.warning(f"Detailed Request Errors: {pop['errors']}")
            time.sleep(10)
            self.status = JobStatus.FAULTED
            return None
        if not pop.get("id"):
            job_skipped_info = pop.get("skipped")
            if job_skipped_info and len(job_skipped_info):
                self.skipped_info = f" Skipped Info: {job_skipped_info}."
            else:
                self.skipped_info = ""
            logger.info(
                f"Server {self.bridge_data.horde_url} has no valid generations to do for us.{self.skipped_info}"
            )
            time.sleep(self.retry_interval)
            self.status = JobStatus.FAULTED
            return None
        return pop

    @logger.catch(reraise=True)
    def start_job(self, pop=None):
        logger.debug("Starting job in threadpool")
        """Starts a job from a pop request"""
        # Pop new request from the Horde
        if pop is None:
            pop = self.get_job_from_server()

        if pop is None:
            logger.error(
                f"Something has gone wrong with {self.bridge_data.horde_url}. Please inform its administrator!"
            )
            time.sleep(self.retry_interval)
            self.status = JobStatus.FAULTED
            return

        self.current_id = pop["id"]
        self.current_payload = pop["payload"]
        self.r2_upload = pop.get("r2_upload", False)
        self.status = JobStatus.WORKING
        # Generate Image
        model = pop.get("model", self.available_models[0])
        self.current_model = model
        # logger.info([self.current_id,self.current_payload])
        use_nsfw_censor = False
        censor_image = None
        censor_reason = None
        if self.bridge_data.censor_nsfw and not self.bridge_data.nsfw:
            use_nsfw_censor = True
            censor_image = self.bridge_data.censor_image_sfw_worker
            censor_reason = "SFW worker"
        censorlist_prompt = self.current_payload["prompt"]
        if "###" in censorlist_prompt:
            censorlist_prompt, _censorlist_negprompt = censorlist_prompt.split("###", 1)
        if any(word in censorlist_prompt for word in self.bridge_data.censorlist):
            use_nsfw_censor = True
            censor_image = self.bridge_data.censor_image_censorlist
            censor_reason = "Censorlist"
        elif self.current_payload.get("use_nsfw_censor", False):
            use_nsfw_censor = True
            censor_image = self.bridge_data.censor_image_sfw_request
            censor_reason = "Requested"
        # use_gfpgan = self.current_payload.get("use_gfpgan", True)
        # use_real_esrgan = self.current_payload.get("use_real_esrgan", False)
        source_processing = pop.get("source_processing")
        source_image = pop.get("source_image")
        source_mask = pop.get("source_mask")
        # These params will always exist in the payload from the horde
        try:
            gen_payload = {
                "prompt": self.current_payload["prompt"],
                "height": self.current_payload["height"],
                "width": self.current_payload["width"],
                "seed": self.current_payload["seed"],
                "n_iter": 1,
                "batch_size": 1,
                "save_individual_images": False,
                "save_grid": False,
            }
            # These params might not always exist in the horde payload
            if "ddim_steps" in self.current_payload:
                gen_payload["ddim_steps"] = self.current_payload["ddim_steps"]
            if "sampler_name" in self.current_payload:
                # K-Diffusers still don't work in our SD2.x models
                gen_payload["sampler_name"] = self.current_payload["sampler_name"]
                if self.model_manager.get_model(model).get("baseline") == "stable diffusion 2":
                    gen_payload["sampler_name"] = "dpmsolver"
            if "cfg_scale" in self.current_payload:
                gen_payload["cfg_scale"] = self.current_payload["cfg_scale"]
            if "ddim_eta" in self.current_payload:
                gen_payload["ddim_eta"] = self.current_payload["ddim_eta"]
            if "denoising_strength" in self.current_payload and source_image:
                gen_payload["denoising_strength"] = self.current_payload["denoising_strength"]
            if self.current_payload.get("karras", False):
                gen_payload["sampler_name"] = gen_payload.get("sampler_name", "k_euler_a") + "_karras"
        except KeyError as err:
            logger.error("Received incomplete payload from job. Aborting. ({})", err)
            self.status = JobStatus.FAULTED
            return
        # logger.debug(gen_payload)
        req_type = "txt2img"
        # TODO: Fix img2img for SD2
        if source_image and self.model_manager.get_model(model).get("baseline") != "stable diffusion 2":
            img_source = None
            img_mask = None
            if source_processing == "img2img":
                req_type = "img2img"
            elif source_processing == "inpainting":
                req_type = "inpainting"
            if source_processing == "outpainting":
                req_type = "outpainting"
        # Prevent inpainting from picking text2img and img2img gens (as those go via compvis pipelines)
        if model == "stable_diffusion_inpainting" and req_type not in [
            "inpainting",
            "outpainting",
        ]:
            # Try to find any other model to do text2img or img2img
            for available_model in self.available_models:
                if available_model != "stable_diffusion_inpainting":
                    model = available_model
            # if the model persists as inpainting for text2img or img2img, we abort.
            if model == "stable_diffusion_inpainting":
                # We remove the base64 from the prompt to avoid flooding the output on the error
                if len(pop.get("source_image", "")) > 10:
                    pop["source_image"] = len(pop.get("source_image", ""))
                if len(pop.get("source_mask", "")) > 10:
                    pop["source_mask"] = len(pop.get("source_mask", ""))
                logger.error(
                    "Received an non-inpainting request for inpainting model. This shouldn't happen. "
                    f"Inform the developer. Current payload {pop}"
                )
                self.status = JobStatus.FAULTED
                return
                # TODO: Send faulted
        logger.debug(f"{req_type} ({model}) request with id {self.current_id} picked up. Initiating work...")
        try:
            safety_checker = (
                self.model_manager.loaded_models["safety_checker"]["model"]
                if "safety_checker" in self.model_manager.loaded_models
                else None
            )

            if source_image:
                base64_bytes = source_image.encode("utf-8")
                img_bytes = base64.b64decode(base64_bytes)
                img_source = Image.open(BytesIO(img_bytes))
            if source_mask:
                base64_bytes = source_mask.encode("utf-8")
                img_bytes = base64.b64decode(base64_bytes)
                img_mask = Image.open(BytesIO(img_bytes))
                if img_mask.size != img_source.size:
                    logger.warning(
                        f"Source image/mask mismatch. Resizing mask from {img_mask.size} to {img_source.size}"
                    )
                    img_mask = img_mask.resize(img_source.size)
        except KeyError:
            self.status = JobStatus.FAULTED
            return
        # If the received image is unreadable, we continue as text2img
        except UnidentifiedImageError:
            logger.error("Source image received for img2img is unreadable. Falling back to text2img!")
            req_type = "txt2img"
            if "denoising_strength" in gen_payload:
                del gen_payload["denoising_strength"]
        except binascii.Error:
            logger.error(
                "Source image received for img2img is cannot be base64 decoded (binascii.Error). "
                "Falling back to text2img!"
            )
            req_type = "txt2img"
            if "denoising_strength" in gen_payload:
                del gen_payload["denoising_strength"]
        if req_type in ["img2img", "txt2img"]:
            if req_type == "img2img":
                gen_payload["init_img"] = img_source
                if img_mask:
                    gen_payload["init_mask"] = img_mask
            generator = CompVis(
                model=self.model_manager.loaded_models[model]["model"],
                device=self.model_manager.loaded_models[model]["device"],
                model_name=model,
                output_dir="bridge_generations",
                load_concepts=True,
                concepts_dir="models/custom/sd-concepts-library",
                safety_checker=safety_checker,
                filter_nsfw=use_nsfw_censor,
                disable_voodoo=self.bridge_data.disable_voodoo.active,
            )
        else:
            if model != "stable_diffusion_inpainting":
                # We remove the base64 from the prompt to avoid flooding the output on the error
                if len(pop.get("source_image", "")) > 10:
                    pop["source_image"] = len(pop.get("source_image", ""))
                if len(pop.get("source_mask", "")) > 10:
                    pop["source_mask"] = len(pop.get("source_mask", ""))
                logger.error(
                    "Received an inpainting request for a non-inpainting model. This shouldn't happen. "
                    f"Inform the developer. Current payload {pop}"
                )
                self.status = JobStatus.FAULTED
                return
            # These variables do not exist in the outpainting implementation
            if "save_grid" in gen_payload:
                del gen_payload["save_grid"]
            if "sampler_name" in gen_payload:
                del gen_payload["sampler_name"]
            if "denoising_strength" in gen_payload:
                del gen_payload["denoising_strength"]
            # We prevent sending an inpainting without mask or transparency, as it will crash us.
            if img_mask is None:
                try:
                    _red, _green, _blue, _alpha = img_source.split()
                except ValueError:
                    logger.warning("inpainting image doesn't have an alpha channel. Aborting gen")
                    self.status = JobStatus.FAULTED
                    return
                    # TODO: Send faulted
            gen_payload["inpaint_img"] = img_source
            if img_mask:
                gen_payload["inpaint_mask"] = img_mask
            generator = inpainting(
                self.model_manager.loaded_models[model]["model"],
                self.model_manager.loaded_models[model]["device"],
                "bridge_generations",
                filter_nsfw=use_nsfw_censor,
                disable_voodoo=self.bridge_data.disable_voodoo.active,
            )
        try:
            logger.debug("Starting generation...")
            generator.generate(**gen_payload)
            logger.debug("Finished generation...")
        except RuntimeError as err:
            stack_payload = gen_payload
            stack_payload["request_type"] = req_type
            stack_payload["model"] = model
            logger.error(
                "Something went wrong when processing request.\n"
                f"Please inform the developers of the below payload:\n{stack_payload}"
            )
            trace = "".join(traceback.format_exception(type(err), err, err.__traceback__))
            logger.trace(trace)
            self.status = JobStatus.FAULTED
            return
        self.image = generator.images[0]["image"]
        self.seed = generator.images[0]["seed"]
        if generator.images[0].get("censored", False):
            logger.debug(f"Image censored with reason: {censor_reason}")
            self.image = censor_image
        logger.debug("censor done...")
        # We unload the generator from RAM
        generator = None
        for post_processor in self.current_payload.get("post_processing", []):
            logger.debug(f"Post-processing with {post_processor}...")
            try:
                self.image = post_process(post_processor, self.image, self.model_manager)
            except (AssertionError, RuntimeError) as err:
                logger.warning(
                    "Post-Processor '{}' encountered an error when working on image . Skipping! {}",
                    post_processor,
                    err,
                )
            if self.r2_upload:
                self.upload_quality = 95
            else:
                if post_processor in ["RealESRGAN_x4plus"]:
                    self.upload_quality = 45
                else:
                    self.upload_quality = 75
        logger.debug("post-processing done...")
        # Not a daemon, so that it can survive after this class is garbage collected
        submit_thread = threading.Thread(target=self.submit_job, args=())
        submit_thread.start()
        logger.debug("Finished job in threadpool")

    def submit_job(self):
        """Submits the job to the server to earn our kudos."""
        self.status = JobStatus.FINALIZING
        # Submit back to horde
        # images, seed, info, stats = txt2img(**self.current_payload)
        buffer = BytesIO()
        # We send as WebP to avoid using all the horde bandwidth
        self.image.save(buffer, format="WebP", quality=self.upload_quality)
        if self.r2_upload:
            put_response = requests.put(self.r2_upload, data=buffer.getvalue())
            generation = "R2"
            logger.debug("R2 Upload response: {}", put_response)
        else:
            generation = base64.b64encode(buffer.getvalue()).decode("utf8")
        self.submit_dict = {
            "id": self.current_id,
            "generation": generation,
            "api_key": self.bridge_data.api_key,
            "seed": self.seed,
            "max_pixels": self.bridge_data.max_pixels,
        }
        while self.is_finalizing():
            if self.loop_retry > 10:
                logger.error(
                    f"Exceeded retry count {self.loop_retry} for generation id {self.current_id}. Aborting generation!"
                )
                self.status = JobStatus.FAULTED
                break
            self.loop_retry += 1
            try:
                logger.debug(
                    f"posting payload with size of {round(sys.getsizeof(json.dumps(self.submit_dict)) / 1024,1)} kb"
                )
                submit_req = requests.post(
                    self.bridge_data.horde_url + "/api/v2/generate/submit",
                    json=self.submit_dict,
                    headers=self.headers,
                    timeout=60,
                )
                logger.debug(f"Upload completed in {submit_req.elapsed.total_seconds()}")
                try:
                    submit = submit_req.json()
                except json.decoder.JSONDecodeError:
                    logger.error(
                        f"Something has gone wrong with {self.bridge_data.horde_url} during submit. "
                        f"Please inform its administrator!  (Retry {self.loop_retry}/10)"
                    )
                    time.sleep(self.retry_interval)
                    continue
                if submit_req.status_code == 404:
                    logger.warning("The generation we were working on got stale. Aborting!")
                    self.status = JobStatus.FAULTED
                    break
                if not submit_req.ok:
                    logger.warning(
                        f"During gen submit, server {self.bridge_data.horde_url} "
                        f"responded with status code {submit_req.status_code}: "
                        f"{submit['message']}. Waiting for 10 seconds...  (Retry {self.loop_retry}/10)"
                    )
                    if "errors" in submit:
                        logger.warning(f"Detailed Request Errors: {submit['errors']}")
                    time.sleep(10)
                    continue
                logger.debug(
                    f'Submitted generation with id {self.current_id} and contributed for {submit_req.json()["reward"]}'
                )
                bridge_stats.update_inference_stats(self.current_model, submit_req.json()["reward"])
                self.status = JobStatus.DONE
                break
            except requests.exceptions.ConnectionError:
                logger.warning(
                    f"Server {self.bridge_data.horde_url} unavailable during submit. "
                    f"Waiting 10 seconds...  (Retry {self.loop_retry}/10)"
                )
                time.sleep(10)
                continue
            except requests.exceptions.ReadTimeout:
                logger.warning(
                    f"Server {self.bridge_data.horde_url} timed out during submit. "
                    f"Waiting 10 seconds...  (Retry {self.loop_retry}/10)"
                )
                time.sleep(10)
                continue
