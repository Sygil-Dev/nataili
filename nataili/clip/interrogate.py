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
import numpy as np
import torch

from nataili.cache import Cache
from nataili.util import logger

from .image import ImageEmbed
from .text import TextEmbed


class Interrogator:
    def __init__(self, model):
        """
        :param model: Loaded model from ModelManager
        For each data list in model["data_lists"], check if all items are cached.
        If not, embed all items and save to cache.
        If yes, load all text embeds from cache.
        """
        self.model = model
        self.cache = Cache(self.model["cache_name"], cache_parentname="embeds", cache_subname="text")
        self.cache_image = Cache(self.model["cache_name"], cache_parentname="embeds", cache_subname="image")
        self.embed_lists = {}
        for k in self.model["data_lists"].keys():
            cached = True
            for text in self.model["data_lists"][k]:
                if text not in self.cache.kv:
                    cached = False
                    break
            if not cached:
                logger.info(f"Caching {k} embeds")
                text_embed = TextEmbed(self.model, self.cache)
                for text in self.model["data_lists"][k]:
                    text_embed(text)
                self.cache.flush()
            else:
                logger.debug(f"{k} embeds already cached")
            logger.debug(f"Loading {k} embeds")
            with torch.no_grad():
                text_features = torch.cat(
                    [
                        torch.from_numpy(np.load(f"{self.cache.cache_dir}/{self.cache.kv[text]}.npy")).float()
                        for text in self.model["data_lists"][k]
                    ],
                    dim=0,
                )
            text_features /= text_features.norm(dim=-1, keepdim=True)
            self.embed_lists[k] = text_features

    def rank(self, image_features, text_array_key, device, top_count=2):
        """
        :param image_features: Image features to compare to text features
        :param text_array_key: Key in model["data_lists"].
            Key is used to get text features from cache.
        :param device: Device to run on
        :param top_count: Number of top results to return
        :return: List of tuples of (text, similarity)
        """
        top_count = min(top_count, len(self.model["data_lists"][text_array_key]))
        text_features = self.embed_lists[text_array_key].to(device)

        similarity = torch.zeros((1, len(self.model["data_lists"][text_array_key]))).to(device)
        for i in range(image_features.shape[0]):
            similarity += (100.0 * image_features[i].unsqueeze(0) @ text_features.T).softmax(dim=-1)
        similarity /= image_features.shape[0]

        top_probs, top_labels = similarity.cpu().topk(top_count, dim=-1)
        return [
            (self.model["data_lists"][text_array_key][top_labels[0][i].numpy()], (top_probs[0][i].numpy() * 100))
            for i in range(top_count)
        ]

    def __call__(self, input_image):
        """
        :param input_image: PIL image
        :return: List of lists of tuples of (text, similarity)
        Input image is embedded and cached, then compared to all text features in model["data_lists"].
        """
        image_embed = ImageEmbed(self.model, self.cache_image)
        image_hash = image_embed(input_image)
        self.cache_image.flush()
        image_embed_array = np.load(f"{self.cache_image.cache_dir}/{self.cache_image.kv[image_hash]}.npy")
        image_features = torch.from_numpy(image_embed_array).float().to(self.model["device"])
        ranks = []
        bests = [[("", 0)]] * 7
        logger.info("Ranking text")
        ranks.append(self.rank(image_features, "mediums", self.model["device"]))
        ranks.append(self.rank(image_features, "flavors", self.model["device"]))
        ranks.append(self.rank(image_features, "artists", self.model["device"]))
        ranks.append(self.rank(image_features, "movements", self.model["device"]))
        ranks.append(self.rank(image_features, "sites", self.model["device"]))
        ranks.append(self.rank(image_features, "techniques", self.model["device"]))
        ranks.append(self.rank(image_features, "tags", self.model["device"]))
        logger.info("Sorting text")
        for i in range(len(ranks)):
            confidence_sum = 0
            for ci in range(len(ranks[i])):
                confidence_sum += ranks[i][ci][1]
            if confidence_sum > sum(bests[i][t][1] for t in range(len(bests[i]))):
                bests[i] = ranks[i]

        for best in bests:
            best.sort(key=lambda x: x[1], reverse=True)

        medium = []
        for m in bests[0][:1]:
            medium.append({"text": m[0], "confidence": m[1]})
        artist = []
        for a in bests[1][:2]:
            artist.append({"text": a[0], "confidence": a[1]})
        trending = []
        for t in bests[2][:2]:
            trending.append({"text": t[0], "confidence": t[1]})
        movement = []
        for m in bests[3][:2]:
            movement.append({"text": m[0], "confidence": m[1]})
        flavors = []
        for f in bests[4][:2]:
            flavors.append({"text": f[0], "confidence": f[1]})
        techniques = []
        for t in bests[5][:2]:
            techniques.append({"text": t[0], "confidence": t[1]})
        tags = []
        for t in bests[6][:2]:
            tags.append({"text": t[0], "confidence": t[1]})
        return {
            "medium": medium,
            "artist": artist,
            "trending": trending,
            "movement": movement,
            "flavors": flavors,
            "techniques": techniques,
            "tags": tags,
        }
