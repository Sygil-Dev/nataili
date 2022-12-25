"""The configuration of the bridge"""
import getpass
import importlib
import os
import random
import sys
import threading

import bridgeData as bd
import requests
from PIL import Image

from nataili import disable_local_ray_temp, disable_voodoo, disable_xformers
from nataili.util import logger
from worker.argparser import args


class BridgeData:
    """Configuration object"""

    def __init__(self):
        random.seed()
        self.horde_url = os.environ.get("HORDE_URL", "https://stablehorde.net")
        # Give a cool name to your instance
        self.worker_name = os.environ.get(
            "HORDE_WORKER_NAME",
            f"Automated Instance #{random.randint(-100000000, 100000000)}",
        )
        # The api_key identifies a unique user in the horde
        self.api_key = os.environ.get("HORDE_API_KEY", "0000000000")
        # Put other users whose prompts you want to prioritize.
        # The owner's username is always included so you don't need to add it here,
        # unless you want it to have lower priority than another user
        self.priority_usernames = list(filter(lambda a: a, os.environ.get("HORDE_PRIORITY_USERNAMES", "").split(",")))
        self.max_power = int(os.environ.get("HORDE_MAX_POWER", 8))
        self.max_threads = int(os.environ.get("HORDE_MAX_THREADS", 1))
        self.queue_size = int(os.environ.get("HORDE_QUEUE_SIZE", 0))
        self.nsfw = os.environ.get("HORDE_NSFW", "true") == "true"
        self.censor_nsfw = os.environ.get("HORDE_CENSOR", "false") == "true"
        self.blacklist = list(filter(lambda a: a, os.environ.get("HORDE_BLACKLIST", "").split(",")))
        self.censorlist = list(filter(lambda a: a, os.environ.get("HORDE_CENSORLIST", "").split(",")))
        self.allow_img2img = os.environ.get("HORDE_IMG2IMG", "true") == "true"
        self.allow_painting = os.environ.get("HORDE_PAINTING", "true") == "true"
        self.allow_unsafe_ip = os.environ.get("HORDE_ALLOW_UNSAFE_IP", "true") == "true"
        self.model_names = os.environ.get("HORDE_MODELNAMES", "stable_diffusion").split(",")
        self.max_pixels = 64 * 64 * 8 * self.max_power
        self.censor_image_sfw_worker = Image.open("assets/nsfw_censor_sfw_worker.png")
        self.censor_image_censorlist = Image.open("assets/nsfw_censor_censorlist.png")
        self.censor_image_sfw_request = Image.open("assets/nsfw_censor_sfw_request.png")
        self.initialized = False
        self.models_reloading = False
        self.username = None
        self.model = None
        self.dynamic_models = True
        self.number_of_dynamic_models = 3
        self.max_models_to_download = 10
        self.models_to_skip = os.environ.get("HORDE_SKIPPED_MODELNAMES", "stable_diffusion_inpainting").split(",")
        self.predefined_models = []

        disable_xformers.toggle(args.disable_xformers)
        disable_local_ray_temp.toggle(args.disable_local_ray_temp)
        disable_voodoo.toggle(args.disable_voodoo)
        if disable_voodoo.active:
            disable_local_ray_temp.activate()

        self.disable_voodoo = disable_voodoo

    @logger.catch(reraise=True)
    def reload_data(self):
        """Reloads configuration data"""
        previous_url = self.horde_url
        previous_api_key = self.api_key
        try:
            # TODO - move this to a yaml file
            importlib.reload(bd)
            self.api_key = bd.api_key
            self.worker_name = bd.worker_name
            self.horde_url = bd.horde_url
            self.priority_usernames = bd.priority_usernames
            self.max_power = bd.max_power
            if not self.dynamic_models:
                self.model_names = bd.models_to_load
            else:
                self.predefined_models = bd.models_to_load
            try:
                self.nsfw = bd.nsfw
            except AttributeError:
                pass
            try:
                self.censor_nsfw = bd.censor_nsfw
            except AttributeError:
                pass
            try:
                self.blacklist = bd.blacklist
            except AttributeError:
                pass
            try:
                self.censorlist = bd.censorlist
            except AttributeError:
                pass
            try:
                self.allow_img2img = bd.allow_img2img
            except AttributeError:
                pass
            try:
                self.allow_painting = bd.allow_painting
            except AttributeError:
                pass
            try:
                self.allow_unsafe_ip = bd.allow_unsafe_ip
            except AttributeError:
                pass
            try:
                self.max_threads = bd.max_threads
            except AttributeError:
                pass
            try:
                self.queue_size = bd.queue_size
            except AttributeError:
                pass
            try:
                self.dynamic_models = bd.dynamic_models
            except AttributeError:
                pass
            try:
                self.number_of_dynamic_models = bd.number_of_dynamic_models
            except AttributeError:
                pass
            try:
                self.max_models_to_download = bd.max_models_to_download
            except AttributeError:
                pass
            try:
                self.models_to_skip = bd.models_to_skip
            except AttributeError:
                pass
        except (ImportError, AttributeError) as err:
            logger.warning("bridgeData.py could not be loaded. Using defaults with anonymous account - {}", err)
        if args.api_key:
            self.api_key = args.api_key
        if args.worker_name:
            self.worker_name = args.worker_name
        if args.horde_url:
            self.horde_url = args.horde_url
        if args.priority_usernames:
            self.priority_usernames = args.priority_usernames
        if args.max_power:
            self.max_power = args.max_power
        if args.max_power:
            self.max_threads = args.max_threads
        if args.queue_size:
            self.queue_size = args.queue_size
        if args.model:
            self.model = [args.model]
        if args.sfw:
            self.nsfw = False
        if args.censor_nsfw:
            self.censor_nsfw = args.censor_nsfw
        if args.blacklist:
            self.blacklist = args.blacklist
        if args.censorlist:
            self.censorlist = args.censorlist
        if args.allow_img2img:
            self.allow_img2img = args.allow_img2img
        if args.allow_painting:
            self.allow_painting = args.allow_painting
        if args.allow_unsafe_ip:
            self.allow_unsafe_ip = args.allow_unsafe_ip
        if args.disable_dynamic_models:
            self.dynamic_models = False
        if self.dynamic_models:
            try:
                from creds import hf_password, hf_username  # noqa F401
            except ImportError:
                logger.warning(
                    "Dynamic models enabled. Please setup creds.py so it won't prompt for authentication later"
                )
        self.max_power = max(self.max_power, 2)
        self.max_pixels = 64 * 64 * 8 * self.max_power
        # if self.censor_nsfw or (self.censorlist is not None and len(self.censorlist)):
        self.model_names.append("safety_checker")
        self.model_names.append("GFPGAN")
        self.model_names.append("RealESRGAN_x4plus")
        # We need to load this dynamically only when we need it
        # self.model_names.append("CodeFormers")
        
        if not self.initialized or previous_api_key != self.api_key:
            try:
                user_req = requests.get(
                    self.horde_url + "/api/v2/find_user",
                    headers={"apikey": self.api_key},
                    timeout=10,
                )
                user_req = user_req.json()
                self.username = user_req["username"]

            # pylint: disable=broad-except
            except Exception:
                logger.warning(f"Server {self.horde_url} error during find_user. Setting username 'N/A'")
                self.username = "N/A"
        if (not self.initialized and not self.models_reloading) or previous_url != self.horde_url:
            logger.init(
                (
                    f"Username '{self.username}'. Server Name '{self.worker_name}'. "
                    f"Horde URL '{self.horde_url}'. Max Pixels {self.max_pixels}"
                ),
                status="Joining Horde",
            )

    @logger.catch(reraise=True)
    def check_models(self, model_manager):
        """Check to see if we have the models needed"""
        if self.models_reloading:
            return
        if not self.initialized:
            logger.init("Models", status="Checking")
        models_exist = True
        not_found_models = []
        for model in self.model_names:
            model_info = model_manager.get_model(model)
            if not model_info:
                logger.warning(
                    f"Model name requested {model} in bridgeData is unknown to us. "
                    "Please check your configuration. Aborting!"
                )
                self.model_names.remove(model)
                continue
            if model in model_manager.get_loaded_models_names():
                continue
            if not model_manager.validate_model(model, skip_checksum=args.skip_md5):
                if (
                    model_manager.count_available_models_by_types() + len(not_found_models)
                    < self.max_models_to_download
                ):
                    models_exist = False
                    not_found_models.append(model)
            # Diffusers library uses its own internal download mechanism
            if model_info["type"] == "diffusers" and model_info["hf_auth"]:
                check_mm_auth(model_manager)
        if not models_exist:
            if args.yes or self.dynamic_models:
                choice = "y"
            else:
                choice = input(
                    "You do not appear to have downloaded the models needed yet.\n"
                    "You need at least a main model to proceed. "
                    f"Would you like to download your prespecified models?\n\
                y: Download {not_found_models} (default).\n\
                n: Abort and exit\n\
                all: Download all basic models (This can take a significant amount of time and bandwidth)\n\
                Please select an option: "
                )
            if choice not in ["y", "Y", "", "yes", "all", "a"]:
                sys.exit(1)
            needs_hf = False
            for model in not_found_models:
                models_to_download = model_manager.get_model_download(model)
                for download_model in models_to_download:
                    if download_model.get("hf_auth", False):
                        needs_hf = True
            if choice in ["all", "a"]:
                needs_hf = True
            if needs_hf:
                check_mm_auth(model_manager)
            model_manager.init()
            model_manager.taint_models(not_found_models)
            if choice in ["all", "a"]:
                model_manager.download_all()
            elif choice in ["y", "Y", "", "yes"]:
                for model in not_found_models:
                    logger.init(f"Model: {model}", status="Downloading")
                    if not model_manager.download_model(model):
                        logger.message(
                            "Something went wrong when downloading the model and it does not fit the expected "
                            "checksum. Please check that your HuggingFace authentication is correct and that "
                            "you've accepted the model license from the browser. This model will be skipped!"
                        )
                        self.model_names.remove(model)
            model_manager.init()
        if not self.initialized:
            logger.init_ok("Models", status="OK")
        if os.path.exists("./bridgeData.py"):
            if not self.initialized:
                logger.init_ok("Bridge Config", status="OK")
        elif input(
            "You do not appear to have a bridgeData.py. Would you like to create it from the template now? (y/n)"
        ) in ["y", "Y", "", "yes"]:
            with open("bridgeData_template.py", "r") as firstfile, open("bridgeData.py", "a") as secondfile:
                for line in firstfile:
                    secondfile.write(line)
            logger.message(
                "bridgeData.py created. Bridge will exit. "
                "Please edit bridgeData.py with your setup and restart the worker"
            )
            sys.exit(2)

    def reload_models(self, model_manager):
        """Reloads models - Note this is IN A THREAD"""
        if self.models_reloading:
            return
        self.models_reloading = True
        thread = threading.Thread(target=self._reload_models, args=(model_manager,))
        thread.daemon = True
        thread.start()

    @logger.catch(reraise=True)
    def _reload_models(self, model_manager):
        for model in model_manager.get_loaded_models_names():
            if model not in self.model_names:
                logger.init(f"{model}", status="Unloading")
                model_manager.unload_model(model)
        for model in self.model_names:
            if model not in model_manager.get_loaded_models_names():
                logger.init(f"{model}", status="Loading")
                success = model_manager.load_model(model)
                if success:
                    logger.init_ok(f"{model}", status="Loaded")
                else:
                    logger.init_err(f"{model}", status="Error")
            self.initialized = True
        self.models_reloading = False


def check_mm_auth(model_manager):
    """Checks for hugging face authentication for model manager"""
    if model_manager.has_authentication():
        return
    if args.hf_token:
        hf_auth = {"username": "USER", "password": args.hf_token}
        model_manager.set_authentication(hf_auth=hf_auth)
        return
    try:
        from creds import hf_password, hf_username
    except ImportError:
        hf_username = input("Please type your huggingface.co username: ")
        hf_password = getpass.getpass("Please type your huggingface.co Access Token or password: ")
    hf_auth = {"username": hf_username, "password": hf_password}
    model_manager.set_authentication(hf_auth=hf_auth)
