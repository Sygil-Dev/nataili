# This file is part of nataili ("Homepage" = "https://github.com/Sygil-Dev/nataili").

# Copyright 2022 hlky
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from functools import wraps
from typing import TypeVar

from torch import amp, dtype, float16, no_grad

T = TypeVar("T")


def autocast_cuda(func: T, dtype: dtype = float16) -> T:
    @wraps(func)
    def wrap(*args, **kwargs):
        return amp.autocast(device_type="cuda", dtype=dtype)(no_grad()(func))(*args, **kwargs)

    return wrap


def autocast_cpu(func: T, dtype: dtype = float16) -> T:
    @wraps(func)
    def wrap(*args, **kwargs):
        return amp.autocast(device_type="cpu", dtype=dtype)(no_grad()(func))(*args, **kwargs)

    return wrap
