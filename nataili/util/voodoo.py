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
import contextlib
import copy
import os
from typing import Dict, List, Tuple, TypeVar

import ray
import torch

from nataili.aitemplate import Model

T = TypeVar("T")


def extract_tensors(m: torch.nn.Module) -> Tuple[torch.nn.Module, List[Dict]]:
    tensors = []
    for _, module in m.named_modules():
        params = {
            name: torch.clone(param).cpu().detach().numpy() for name, param in module.named_parameters(recurse=False)
        }
        buffers = {name: torch.clone(buf).cpu().detach().numpy() for name, buf in module.named_buffers(recurse=False)}
        tensors.append({"params": params, "buffers": buffers})

    m_copy = copy.deepcopy(m)
    for _, module in m_copy.named_modules():
        for name in [name for name, _ in module.named_parameters(recurse=False)] + [
            name for name, _ in module.named_buffers(recurse=False)
        ]:
            setattr(module, name, None)

    m_copy.train(False)
    return m_copy, tensors


def replace_tensors(m: torch.nn.Module, tensors: List[Dict], device="cuda"):
    modules = [module for _, module in m.named_modules()]
    for module, tensor_dict in zip(modules, tensors):
        for name, array in tensor_dict["params"].items():
            module.register_parameter(
                name,
                torch.nn.Parameter(torch.as_tensor(array, device=device), requires_grad=False),
            )
        for name, array in tensor_dict["buffers"].items():
            module.register_buffer(name, torch.as_tensor(array, device=device))


@contextlib.contextmanager
def load_from_plasma(ref, device="cuda"):
    skeleton, weights = ray.get(ref)
    replace_tensors(skeleton, weights, device=device)
    skeleton.eval().to(device, memory_format=torch.channels_last)
    yield skeleton
    torch.cuda.empty_cache()


def push_model_to_plasma(model: torch.nn.Module) -> ray.ObjectRef:
    ref = ray.put(extract_tensors(model))
    return ref


@contextlib.contextmanager
def load_diffusers_pipeline_from_plasma(ref, device="cuda"):
    pipe, modules = ray.get(ref)
    for name, weights in modules.items():
        replace_tensors(getattr(pipe, name), weights, device=device)
    pipe.to(device)
    yield pipe
    torch.cuda.empty_cache()


def push_diffusers_pipeline_to_plasma(pipe) -> ray.ObjectRef:
    modules = {}
    components = pipe.components
    for name, component in components.items():
        if isinstance(component, torch.nn.Module):
            skeleton, weights = extract_tensors(component)
            setattr(pipe, name, skeleton)
            modules[name] = weights
    ref = ray.put((pipe, modules))
    return ref


def init_ait_module(
    model_name,
    workdir,
):
    mod = Model(os.path.join(workdir, model_name))
    return mod


def push_ait_module(module: Model) -> ray.ObjectRef:
    ref = ray.put(module)
    return ref


def load_ait_module(ref):
    ait_module = ray.get(ref)
    return ait_module
