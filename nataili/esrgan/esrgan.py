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
from PIL import Image

from nataili.util import PostProcessor


class esrgan(PostProcessor):
    def set_filename_append(self):
        """
        Set filename append
        """
        self.filename_append = "esrgan"

    def process(self, img, img_array, **kwargs):
        """
        Override process method from PostProcessor
        :param img: PIL Image
        :param img_array: numpy array
        :param kwargs: strength
        :return: PIL Image
        """
        output, _ = self.model["model"].enhance(img_array)
        output_array = np.array(output)
        output_image = Image.fromarray(output_array)
        return output_image
