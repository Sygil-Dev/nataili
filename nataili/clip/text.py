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
from uuid import uuid4

import clip
import numpy as np
import torch

from nataili.cache import Cache
from nataili.util import autocast_cuda


class TextEmbed:
    def __init__(self, model, cache: Cache):
        """
        :param model: Loaded model from ModelManager
        :param cache: Cache object
        """
        self.model = model
        self.cache = cache

    @autocast_cuda
    def __call__(self, text):
        """
        :param text: Text to embed
        If text is not in cache, embed it and save it to cache
        """
        if text in self.cache.kv:
            return
        text_tokens = clip.tokenize([text]).cuda()
        with torch.no_grad():
            text_features = self.model["model"].encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_embed_array = text_features.cpu().detach().numpy()
        filename = str(uuid4())
        np.save(f"{self.cache.cache_dir}/{filename}", text_embed_array)
        self.cache.kv[text] = filename
