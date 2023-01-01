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
from .aitemplate import AITemplateModelManager
from .blip import BlipModelManager
from .clip import ClipModelManager
from .codeformer import CodeFormerModelManager
from .compvis import CompVisModelManager
from .esrgan import EsrganModelManager
from .gfpgan import GfpganModelManager


class ModelManager:
    def __init__(self):
        self.aitemplate = AITemplateModelManager()
        self.blip = BlipModelManager()
        self.clip = ClipModelManager()
        self.compvis = CompVisModelManager()
        self.esrgan = EsrganModelManager()
        self.gfpgan = GfpganModelManager()
        self.codeformer = CodeFormerModelManager(self.gfpgan, self.esrgan)
