"""
This file is part of nataili ("Homepage" = "https://github.com/Sygil-Dev/nataili").

Copyright 2022 hlky and Sygil-Dev
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from uuid import uuid4

import git
import requests
import torch
from tqdm import tqdm
from transformers import logging

from ..util import logger

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

logging.set_verbosity_error()


class BaseModelManager:
    def __init__(self):
        self.path = f"{Path.home()}/.cache/nataili/"
        self.models = {}
        self.available_models = []
        self.loaded_models = {}
        self.pkg = importlib_resources.files("nataili")
        self.models_path = self.pkg / "models.json"
        self.cuda_available = torch.cuda.is_available()
        self.cuda_devices, self.recommended_gpu = self.get_cuda_devices()

    def init(self):
        self.models = json.loads((self.models_path).read_text())
        models_available = []
        for model in self.models:
            if self.check_model_available(model):
                models_available.append(model)
        self.available_models = models_available

    def get_model_files(self, model_name):
        """
        :param model_name: Name of the model
        Returns the files for a model
        """
        return self.models[model_name]["config"]["files"]

    def get_model_download(self, model_name):
        """
        :param model_name: Name of the model
        Returns the download details for a model
        """
        return self.models[model_name]["config"]["download"]

    def get_available_models(self):
        """
        Returns the available models
        """
        return self.available_models

    def get_available_models_by_types(self, model_types=None):
        if not model_types:
            model_types = ["ckpt", "diffusers"]
        models_available = []
        for model in self.models:
            if self.models[model]["type"] in model_types and self.check_available(self.get_model_files(model)):
                models_available.append(model)
        return models_available

    def count_available_models_by_types(self, model_types=None):
        return len(self.get_available_models_by_types(model_types))

    def get_loaded_models(self):
        """
        Returns the loaded models
        """
        return self.loaded_models

    def get_loaded_model(self, model_name):
        """
        :param model_name: Name of the model
        Returns the loaded model
        """
        return self.loaded_models[model_name]

    def get_loaded_models_names(self, string=False):
        """
        :param string: If True, returns concatenated string of model names
        Returns a list of the loaded model names
        """
        if string:
            return ", ".join(self.loaded_models.keys())
        return list(self.loaded_models.keys())

    def is_model_loaded(self, model_name):
        """
        :param model_name: Name of the model
        Returns whether the model is loaded
        """
        return model_name in self.loaded_models

    def unload_model(self, model_name):
        """
        :param model_name: Name of the model
        Unloads a model
        """
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            return True
        return False

    def unload_all_models(self):
        """
        Unloads all models
        """
        for model in self.loaded_models:
            del self.loaded_models[model]
        return True

    def taint_model(self, model_name):
        """Marks a model as not valid by remiving it from available_models"""
        if model_name in self.available_models:
            self.available_models.remove(model_name)
            self.tainted_models.append(model_name)

    def taint_models(self, models):
        for model in models:
            self.taint_model(model)

    def validate_model(self, model_name, skip_checksum=False):
        """
        :param model_name: Name of the model
        :param skip_checksum: If True, skips checksum validation
        For each file in the model, checks if the file exists and if the checksum is correct
        Returns True if all files are valid, False otherwise
        """
        files = self.get_model_files(model_name)
        for file_details in files:
            if not self.check_file_available(file_details["path"]):
                return False
            if not skip_checksum and not self.validate_file(file_details):
                return False
        return True

    def validate_file(self, file_details):
        """
        :param file_details: A single file from the model's files list
        Checks if the file exists and if the checksum is correct
        Returns True if the file is valid, False otherwise
        """
        if "md5sum" in file_details:
            full_path = f"{self.path}/{file_details['path']}"
            logger.debug(f"Getting md5sum of {full_path}")
            with open(full_path, "rb") as file_to_check:
                file_hash = hashlib.md5()
                while True:
                    chunk = file_to_check.read(8192)  # Changed just because it broke pylint
                    if not chunk:
                        break
                    file_hash.update(chunk)
            if file_details["md5sum"] != file_hash.hexdigest():
                return False
        return True

    def check_file_available(self, file_path):
        """
        :param file_path: Path of the model's file. File is from the model's files list.
        Checks if the file exists
        Returns True if the file exists, False otherwise
        """
        full_path = f"{self.path}/{file_path}"
        return os.path.exists(full_path)

    def check_available(self, files):
        """
        :param files: List of files from the model's files list
        Checks if all files exist
        Returns True if all files exist, False otherwise
        """
        available = True
        for file in files:
            if not self.check_file_available(file["path"]):
                available = False
        return available

    def download_file(self, url, file_path):
        """
        :param url: URL of the file to download. URL is from the model's download list.
        :param file_path: Path of the model's file. File is from the model's files list.
        Downloads a file
        """
        full_path = f"{self.path}/{file_path}"
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        pbar_desc = full_path.split("/")[-1]
        r = requests.get(url, stream=True, allow_redirects=True)
        with open(full_path, "wb") as f:
            with tqdm(
                # all optional kwargs
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                desc=pbar_desc,
                total=int(r.headers.get("content-length", 0)),
            ) as pbar:
                for chunk in r.iter_content(chunk_size=16 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    def download_model(self, model_name):
        """
        :param model_name: Name of the model
        Checks if the model is available, downloads the model if it is not available.
        After download, validates the model.
        Returns True if the model is available, False otherwise.

        Supported download types:
        - http(s) (url)
        - git (repo url)

        Other:
        - write content to file
        - symlink file
        - delete file
        - unzip file
        """
        if model_name in self.available_models:
            logger.info(f"{model_name} is already available.")
            return True
        download = self.get_model_download(model_name)
        files = self.get_model_files(model_name)
        for i in range(len(download)):
            file_path = (
                f"{download[i]['file_path']}/{download[i]['file_name']}"
                if "file_path" in download[i]
                else files[i]["path"]
            )

            if "file_url" in download[i]:
                download_url = download[i]["file_url"]
            if "file_name" in download[i]:
                download_name = download[i]["file_name"]
            if "file_path" in download[i]:
                download_path = download[i]["file_path"]

            if "manual" in download[i]:
                logger.warning(
                    f"The model {model_name} requires manual download from {download_url}. "
                    f"Please place it in {download_path}/{download_name} then press ENTER to continue..."
                )
                input("")
                continue
            # TODO: simplify
            if "file_content" in download[i]:
                file_content = download[i]["file_content"]
                logger.info(f"writing {file_content} to {file_path}")
                os.makedirs(download_path, exist_ok=True)
                with open(os.path.join(download_path, download_name), "w") as f:
                    f.write(file_content)
            elif "symlink" in download[i]:
                logger.info(f"symlink {file_path} to {download[i]['symlink']}")
                symlink = download[i]["symlink"]
                os.makedirs(download_path, exist_ok=True)
                os.symlink(symlink, os.path.join(download_path, download_name))
            elif "git" in download[i]:
                logger.info(f"git clone {download_url} to {file_path}")
                os.makedirs(file_path, exist_ok=True)
                git.Git(file_path).clone(download_url)
                if "post_process" in download[i]:
                    for post_process in download[i]["post_process"]:
                        if "delete" in post_process:
                            logger.info(f"delete {post_process['delete']}")
                            try:
                                shutil.rmtree(post_process["delete"])
                            except PermissionError as e:
                                logger.error(
                                    f"[!] Something went wrong while deleting the `{post_process['delete']}`. "
                                    "Please delete it manually."
                                )
                                logger.error("PermissionError: ", e)
            elif "unzip" in download[i]:
                zip_path = f"{self.path}/{download_name}.zip"
                temp_path = f"{self.path}/{str(uuid4())}/"
                os.makedirs(temp_path, exist_ok=True)
                self.download_file(download_url, zip_path)
                logger.info(f"unzip {zip_path}")
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(temp_path)
                logger.info(f"moving {temp_path} to {download_path}")
                shutil.move(temp_path, download_path)
                logger.info(f"delete {zip_path}")
                os.remove(zip_path)
                logger.info(f"delete {temp_path}")
                shutil.rmtree(temp_path)
            else:
                if not self.check_file_available(file_path):
                    logger.debug(f"Downloading {download_url} to {file_path}")
                    self.download_file(download_url, file_path)
        if not self.validate_model(model_name):
            return False
        self.init()
        return True

    def download_all_models(self):
        """
        Downloads all models
        """
        for model in self.get_filtered_model_names(download_all=True):
            if not self.check_model_available(model):
                logger.init(f"{model}", status="Downloading")
                self.download_model(model)
            else:
                logger.info(f"{model} is already downloaded.")
        return True

    def check_model_available(self, model_name):
        """
        :param model_name: Name of the model
        Checks if the model is available.
        Returns True if the model is available, False otherwise.
        """
        if model_name not in self.models:
            return False
        return self.check_available(self.get_model_files(model_name))

    def get_cuda_devices(self):
        """
        Checks if CUDA is available.
        If CUDA is available, it returns a list of all available CUDA devices.
        If CUDA is not available, it returns an empty list.
        CUDA Device info: id, name, sm
        List is sorted by sm (compute capability) in descending order.
        Also returns the recommended GPU (highest sm).
        """

        if torch.cuda.is_available():
            number_of_cuda_devices = torch.cuda.device_count()
            cuda_arch = []
            for i in range(number_of_cuda_devices):
                cuda_device = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "sm": torch.cuda.get_device_capability(i)[0] * 10 + torch.cuda.get_device_capability(i)[1],
                }
                cuda_arch.append(cuda_device)
            cuda_arch = sorted(cuda_arch, key=lambda k: k["sm"], reverse=True)
            recommended_gpu = [x for x in cuda_arch if x["sm"] == cuda_arch[0]["sm"]]
            return cuda_arch, recommended_gpu
        else:
            return None

    def get_filtered_models(self, **kwargs):
        """
        Get all models
        :param kwargs: filter based on metadata of the model reference db
        :return: list of models
        """
        filtered_models = self.models
        for keyword in kwargs:
            iterating_models = filtered_models.copy()
            filtered_models = {}
            for model in iterating_models:
                if iterating_models[model].get(keyword) == kwargs[keyword]:
                    filtered_models[model] = iterating_models[model]
        return filtered_models

    def get_filtered_model_names(self, **kwargs):
        """
        Get all model names
        :param kwargs: filter based on metadata of the model reference db
        :return: list of model names
        """
        filtered_models = self.get_filtered_models(**kwargs)
        return list(filtered_models.keys())
