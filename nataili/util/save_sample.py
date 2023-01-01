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
import os


def save_sample(
    image,
    filename,
    sample_path,
    extension="png",
    jpg_quality=95,
    webp_quality=95,
    webp_lossless=True,
    png_compression=9,
):
    path = os.path.join(sample_path, filename + "." + extension)
    if os.path.exists(path):
        return False
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    if extension == "png":
        image.save(path, format="PNG", compress_level=png_compression)
    elif extension == "jpg":
        image.save(path, quality=jpg_quality, optimize=True)
    elif extension == "webp":
        image.save(path, quality=webp_quality, lossless=webp_lossless)
    else:
        return False
    if os.path.exists(path):
        return True
    else:
        return False
