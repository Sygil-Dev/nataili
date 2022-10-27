import requests, json, os, time, argparse, urllib3, time,base64,re

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-i', '--interval', action="store", required=False, type=int, default=1, help="The amount of seconds with which to check if there's new prompts to generate")
arg_parser.add_argument('-a','--api_key', action="store", required=False, type=str, help="The API key corresponding to the owner of this Horde instance")
arg_parser.add_argument('-n','--worker_name', action="store", required=False, type=str, help="The server name for the Horde. It will be shown to the world and there can be only one.")
arg_parser.add_argument('-u','--horde_url', action="store", required=False, type=str, help="The SH Horde URL. Where the bridge will pickup prompts and send the finished generations.")
arg_parser.add_argument('--priority_usernames',type=str, action='append', required=False, help="Usernames which get priority use in this horde instance. The owner's username is always in this list.")
arg_parser.add_argument('-p','--max_power',type=int, required=False, help="How much power this instance has to generate pictures. Min: 2")
arg_parser.add_argument('--sfw', action='store_true', required=False, help="Set to true if you do not want this worker generating NSFW images.")
arg_parser.add_argument('--blacklist', nargs='+', required=False, help="List the words that you want to blacklist.")
arg_parser.add_argument('--censorlist', nargs='+', required=False, help="List the words that you want to censor.")
arg_parser.add_argument('--censor_nsfw', action='store_true', required=False, help="Set to true if you want this bridge worker to censor NSFW images.")
arg_parser.add_argument('--allow_img2img', action='store_true', required=False, help="Set to true if you want this bridge worker to allow img2img request.")
arg_parser.add_argument('--allow_unsafe_ip', action='store_true', required=False, help="Set to true if you want this bridge worker to allow img2img requests from unsafe IPs.")
arg_parser.add_argument('-m', '--model', action='store', required=False, help="Which model to run on this horde.")
arg_parser.add_argument('--debug', action="store_true", default=False, help="Show debugging messages.")
arg_parser.add_argument('-v', '--verbosity', action='count', default=0, help="The default logging level is ERROR or higher. This value increases the amount of logging seen in your screen")
arg_parser.add_argument('-q', '--quiet', action='count', default=0, help="The default logging level is ERROR or higher. This value decreases the amount of logging seen in your screen")
arg_parser.add_argument('--log_file', action='store_true', default=False, help="If specified will dump the log to the specified file")
args = arg_parser.parse_args()

from nataili.inference.compvis.img2img import img2img
from nataili.model_manager import ModelManager
from nataili.inference.compvis.txt2img import txt2img
from nataili.util.cache import torch_gc
from nataili.util import logger,set_logger_verbosity, quiesce_logger, test_logger
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps, ImageChops, UnidentifiedImageError
from io import BytesIO
from base64 import binascii

import random
model = ''
max_content_length = 1024
max_length = 80
current_softprompt = None
softprompts = {}
import os

class BridgeData(object):
    def __init__(self):
        random.seed()
        self.horde_url =os.environ.get("HORDE_URL", "https://stablehorde.net")
        # Give a cool name to your instance
        self.worker_name = os.environ.get("HORDE_WORKER_NAME", f"Automated Instance #{random.randint(-100000000, 100000000)}")
        # The api_key identifies a unique user in the horde
        self.api_key = os.environ.get("HORDE_API_KEY", "0000000000")
        # Put other users whose prompts you want to prioritize.
        # The owner's username is always included so you don't need to add it here, unless you want it to have lower priority than another user
        self.priority_usernames =  list(filter(lambda a : a,os.environ.get("HORDE_PRIORITY_USERNAMES", "").split(",")))
        self.max_power = int(os.environ.get("HORDE_MAX_POWER", 8))
        self.nsfw = os.environ.get("HORDE_NSFW", "true") == "true"
        self.censor_nsfw =  os.environ.get("HORDE_CENSOR", "false") == "true"
        self.blacklist = list(filter(lambda a : a,os.environ.get("HORDE_BLACKLIST", "").split(",")))
        self.censorlist =  list(filter(lambda a : a,os.environ.get("HORDE_CENSORLIST", "").split(",")))
        self.allow_img2img = os.environ.get("HORDE_IMG2IMG", "true") == "true"
        self.allow_unsafe_ip = os.environ.get("HORDE_ALLOW_UNSAFE_IP", "true") == "true"
        self.model_names = os.environ.get("HORDE_MODELNAMES", "stable_diffusion").split(",")
        self.max_pixels = 64*64*8*self.max_power



@logger.catch(reraise=True)
def bridge(interval, model_manager, bd):
    horde_url = bd.horde_url # Will replace later
    current_id = None
    current_payload = None
    loop_retry = 0
    while True:
        if loop_retry > 10 and current_id:
            logger.error(f"Exceeded retry count {loop_retry} for generation id {current_id}. Aborting generation!")
            current_id = None
            current_payload = None
            current_generation = None
            loop_retry = 0
        elif current_id:
            logger.debug(f"Retrying ({loop_retry}/10) for generation id {current_id}...")
        available_models = model_manager.get_loaded_models_names()
        gen_dict = {
            "name": bd.worker_name,
            "max_pixels": bd.max_pixels,
            "priority_usernames": bd.priority_usernames,
            "nsfw": bd.nsfw,
            "blacklist": bd.blacklist,
            "models": available_models,
            "allow_img2img": bd.allow_img2img,
            "allow_unsafe_ip": bd.allow_unsafe_ip,
            "bridge_version": 3,
        }
        # logger.debug(gen_dict)
        headers = {"apikey": bd.api_key}
        if current_id:
            loop_retry += 1
        else:
            try:
                pop_req = requests.post(horde_url + '/api/v2/generate/pop', json = gen_dict, headers = headers)
            except requests.exceptions.ConnectionError:
                logger.warning(f"Server {horde_url} unavailable during pop. Waiting 10 seconds...")
                time.sleep(10)
                continue
            except TypeError:
                logger.warning(f"Server {horde_url} unavailable during pop. Waiting 10 seconds...")
                time.sleep(2)
                continue
            try:
                pop = pop_req.json()
            except json.decoder.JSONDecodeError:
                logger.error(f"Could not decode response from {horde_url} as json. Please inform its administrator!")
                time.sleep(interval)
                continue
            if pop == None:
                logger.error(f"Something has gone wrong with {horde_url}. Please inform its administrator!")
                time.sleep(interval)
                continue
            if not pop_req.ok:
                message = pop['message']
                logger.warning(f"During gen pop, server {horde_url} responded with status code {pop_req.status_code}: {pop['message']}. Waiting for 10 seconds...")
                if 'errors' in pop:
                    logger.warning(f"Detailed Request Errors: {pop['errors']}")
                time.sleep(10)
                continue
            if not pop.get("id"):
                skipped_info = pop.get('skipped')
                if skipped_info and len(skipped_info):
                    skipped_info = f" Skipped Info: {skipped_info}."
                else:
                    skipped_info = ''
                logger.debug(f"Server {horde_url} has no valid generations to do for us.{skipped_info}")
                time.sleep(interval)
                continue
            current_id = pop['id']
            current_payload = pop['payload']
        model = pop.get("model", available_models[0])
        # logger.info([current_id,current_payload])
        use_nsfw_censor = current_payload.get("use_nsfw_censor", False)
        if bd.censor_nsfw and not bd.nsfw:
            use_nsfw_censor = True
        elif any(word in current_payload['prompt'] for word in bd.censorlist):
            use_nsfw_censor = True
        use_gfpgan = current_payload.get("use_gfpgan", True)
        use_real_esrgan = current_payload.get("use_real_esrgan", False)
        source_image = pop.get("source_image")
        # These params will always exist in the payload from the horde
        gen_payload = {
            "prompt": current_payload["prompt"],
            "height": current_payload["height"],
            "width": current_payload["width"],
            "width": current_payload["width"],
            "seed": current_payload["seed"],
            "n_iter": 1,
            "batch_size": 1,
            "save_individual_images": False,
            "save_grid": False,
        }
        # These params might not always exist in the horde payload
        if 'ddim_steps' in current_payload: gen_payload['ddim_steps'] = current_payload['ddim_steps']
        if 'sampler_name' in current_payload: gen_payload['sampler_name'] = current_payload['sampler_name']
        if 'cfg_scale' in current_payload: gen_payload['cfg_scale'] = current_payload['cfg_scale']
        if 'ddim_eta' in current_payload: gen_payload['ddim_eta'] = current_payload['ddim_eta']
        if 'denoising_strength' in current_payload and source_image: 
            gen_payload['denoising_strength'] = current_payload['denoising_strength']
        # logger.debug(gen_payload)
        req_type = "txt2img"
        if source_image:
            req_type = "img2img"
        logger.debug(f"{req_type} ({model}) request with id {current_id} picked up. Initiating work...")
        try:
            safety_checker = model_manager['safety_checker']['model'] if 'safety_checker' in model_manager.loaded_models else None
            if source_image:
                base64_bytes = source_image.encode('utf-8')
                img_bytes = base64.b64decode(base64_bytes)
                gen_payload['init_img'] = Image.open(BytesIO(img_bytes))
                generator = img2img(model_manager.loaded_models[model]["model"], model_manager.loaded_models[model]["device"], 'bridge_generations',
                load_concepts=True, concepts_dir='models/custom/sd-concepts-library', safety_checker=safety_checker, filter_nsfw=use_nsfw_censor)
            else:
                generator = txt2img(model_manager.loaded_models[model]["model"], model_manager.loaded_models[model]["device"], 'bridge_generations',
                load_concepts=True, concepts_dir='models/custom/sd-concepts-library', safety_checker=safety_checker, filter_nsfw=use_nsfw_censor)
        except KeyError:
            continue
        # If the received image is unreadable, we continue
        except UnidentifiedImageError:
            logger.error(f"Source image received for img2img is unreadable. Falling back to text2img!")
            if 'denoising_strength' in gen_payload:
                del gen_payload['denoising_strength']
            generator = txt2img(model_manager.loaded_models[model]["model"], model_manager.loaded_models[model]["device"], 'bridge_generations', load_concepts=True, concepts_dir='models/custom/sd-concepts-library')
        except binascii.Error:
            logger.error(f"Source image received for img2img is cannot be base64 decoded (binascii.Error). Falling back to text2img!")
            if 'denoising_strength' in gen_payload:
                del gen_payload['denoising_strength']
            generator = txt2img(model_manager.loaded_models[model]["model"], model_manager.loaded_models[model]["device"], 'bridge_generations', load_concepts=True, concepts_dir='models/custom/sd-concepts-library')
        generator.generate(**gen_payload)
        torch_gc()
      


        # images, seed, info, stats = txt2img(**current_payload)
        buffer = BytesIO()
        # We send as WebP to avoid using all the horde bandwidth
        image = generator.images[0]["image"]
        seed = generator.images[0]["seed"]
        image.save(buffer, format="WebP", quality=90)
        # logger.info(info)
        submit_dict = {
            "id": current_id,
            "generation": base64.b64encode(buffer.getvalue()).decode("utf8"),
            "api_key": bd.api_key,
            "seed": seed,
            "max_pixels": bd.max_pixels,
        }
        current_generation = seed
        while current_id and current_generation != None:
            try:
                submit_req = requests.post(horde_url + '/api/v2/generate/submit', json = submit_dict, headers = headers)
                try:
                    submit = submit_req.json()
                except json.decoder.JSONDecodeError:
                    logger.error(f"Something has gone wrong with {horde_url} during submit. Please inform its administrator!  (Retry {loop_retry}/10)")
                    time.sleep(interval)
                    continue
                if submit_req.status_code == 404:
                    logger.warning(f"The generation we were working on got stale. Aborting!")
                elif not submit_req.ok:
                    logger.warning(f"During gen submit, server {horde_url} responded with status code {submit_req.status_code}: {submit['message']}. Waiting for 10 seconds...  (Retry {loop_retry}/10)")
                    if 'errors' in submit:
                        logger.warning(f"Detailed Request Errors: {submit['errors']}")
                    time.sleep(10)
                    continue
                else:
                    logger.info(f'Submitted generation with id {current_id} and contributed for {submit_req.json()["reward"]}')
                current_id = None
                current_payload = None
                current_generation = None
                loop_retry = 0
            except requests.exceptions.ConnectionError:
                logger.warning(f"Server {horde_url} unavailable during submit. Waiting 10 seconds...  (Retry {loop_retry}/10)")
                time.sleep(10)
                continue
        time.sleep(interval)

@logger.catch(reraise=True)
def check_models(models):
    logger.init("Models", status="Checking")
    from os.path import exists
    import sys
    mm = ModelManager()
    models_exist = True
    not_found_models = []
    for model in models:
        if not mm.get_model(model):
            logger.error(f"Model name requested {model} in bridgeData is unknown to us. Please check your configuration. Aborting!")
            sys.exit(1)
        if not mm.validate_model(model):
            models_exist = False
            not_found_models.append(model)
    if not models_exist:
        choice = input(f"You do not appear to have downloaded the models needed yet.\nYou need at least a main model to proceed. Would you like to download your prespecified models?\n\
        y: Download {not_found_models} (default).\n\
        n: Abort and exit\n\
        all: Download all models (This can take a significant amount of time and bandwidth)?\n\
        Please select an option: ")
        if choice not in ['y', 'Y', '', 'yes', 'all', 'a']:
            sys.exit(1)
        needs_hf = False
        for model in not_found_models:
            dl = mm.get_model_download(model)
            for m in dl:
                if m.get('hf_auth', False):
                    needs_hf = True
        if needs_hf or choice in ['all', 'a']:
            try:
                from creds import hf_username,hf_password
            except:
                hf_username = input("Please type your huggingface.co username: ")
                hf_password = input("Please type your huggingface.co Access Token or password: ")
            hf_auth = {"username": hf_username, "password": hf_password}
            mm.set_authentication(hf_auth=hf_auth)
        mm.init()
        mm.taint_models(not_found_models)
        if choice in ['all', 'a']:
            mm.download_all()    
        elif choice in ['y', 'Y', '', 'yes']:
            for model in not_found_models:
                logger.init(f"Model: {model}", status="Downloading")
                if not mm.download_model(model):
                    logger.message("Something went wrong when downloading the model and it does not fit the expected checksum. Please check that your HuggingFace authentication is correct and that you've accepted the model license from the browser.")
                    sys.exit(0)
    logger.init_ok("Models", status="OK")
    if exists('./bridgeData.py'):
        logger.init_ok("Bridge Config", status="OK")
    elif input("You do not appear to have a bridgeData.py. Would you like to create it from the template now? (y/n)") in ['y', 'Y', '', 'yes']:
        with open('bridgeData_template.py','r') as firstfile, open('bridgeData.py','a') as secondfile:
            for line in firstfile:
                secondfile.write(line)
        logger.message("bridgeData.py created. Bridge will exit. Please edit bridgeData.py with your setup and restart the bridge")
        sys.exit(0)
    
def load_bridge_data():
    bridge_data = BridgeData()
    try:
        import bridgeData as bd
        bridge_data.api_key = bd.api_key
        bridge_data.worker_name = bd.worker_name
        bridge_data.horde_url = bd.horde_url
        bridge_data.priority_usernames = bd.priority_usernames
        bridge_data.max_power = bd.max_power
        bridge_data.model_names = bd.models_to_load
        try:
            bridge_data.nsfw = bd.nsfw 
        except AttributeError:
            pass
        try:
            bridge_data.censor_nsfw = bd.censor_nsfw
        except AttributeError:
            pass
        try:
            bridge_data.blacklist = bd.blacklist
        except AttributeError:
            pass
        try:
            bridge_data.censorlist = bd.censorlist
        except AttributeError:
            pass
        try:
            bridge_data.allow_img2img = bd.allow_img2img
        except AttributeError:
            pass
        try:
            bridge_data.allow_unsafe_ip = bd.allow_unsafe_ip
        except AttributeError:
            pass
    except:
        logger.warning("bridgeData.py could not be loaded. Using defaults with anonymous account")
    if args.api_key: bridge_data.api_key = args.api_key 
    if args.worker_name: bridge_data.worker_name = args.worker_name 
    if args.horde_url: bridge_data.horde_url = args.horde_url 
    if args.priority_usernames: bridge_data.priority_usernames = args.priority_usernames 
    if args.max_power: bridge_data.max_power = args.max_power 
    if args.model: bridge_data.model = [args.model] 
    if args.sfw: bridge_data.nsfw = False
    if args.censor_nsfw: bridge_data.censor_nsfw = args.censor_nsfw
    if args.blacklist: bridge_data.blacklist = args.blacklist
    if args.censorlist: bridge_data.censorlist = args.censorlist
    if args.allow_img2img: bridge_data.allow_img2img = args.allow_img2img
    if args.allow_unsafe_ip: bridge_data.allow_unsafe_ip = args.allow_unsafe_ip
    if bridge_data.max_power < 2:
        bridge_data.max_power = 2
    bridge_data.max_pixels = 64*64*8*bridge_data.max_power
    return(bridge_data)

if __name__ == "__main__":
    
    set_logger_verbosity(args.verbosity)
    if args.log_file:
        logger.add("koboldai_bridge_log.log", retention="7 days", level="warning")    # Automatically rotate too big file
    quiesce_logger(args.quiet)
    bd = load_bridge_data()
    # test_logger()
    check_models(bd.model_names)
    model_manager = ModelManager()
    model_manager.init()
    if args.censor_nsfw or bd.censor_nsfw:
        model = 'safety_checker'
        while model not in model_manager.available_models:
            logger.warning(f"Model {model} is not available. Downloading it.")
            model_manager.download_model(model)
        logger.init(f'{model}', status="Loading")
        success = model_manager.load_model(model)
        if success:
            logger.init_ok(f'{model}', status="Loaded")
        else:
            logger.init_err(f'{model}', status="Error")
    for model in bd.model_names:
        logger.init(f'{model}', status="Loading")
        success = model_manager.load_model(model)
        if success:
            logger.init_ok(f'{model}', status="Loaded")
        else:
            logger.init_err(f'{model}', status="Error")
    logger.init(f"API Key '{bd.api_key}'. Server Name '{bd.worker_name}'. Horde URL '{bd.horde_url}'. Max Pixels {bd.max_pixels}", status="Joining Horde")
    try:
        bridge(args.interval, model_manager, bd)
    except KeyboardInterrupt:
        logger.info(f"Keyboard Interrupt Received. Ending Process")
    logger.init(f"{bd.worker_name} Instance", status="Stopped")
