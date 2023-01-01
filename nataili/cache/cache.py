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
import json
import os
from pathlib import Path


class Cache:
    def __init__(self, cache_name, cache_subname=None, cache_parentname=None):
        """
        :param cache_name: Name of the cache
        :param cache_subname: Subfolder in the cache
        :param cache_parentname: Parent folder of the cache
        Examples:
        cache = Cache("test", "sub", "parent")
        path = self.path + "/parent/test/sub"

        cache = Cache("test", "sub")
        path = self.path + "/test/sub"

        cache = Cache("test")
        path = self.path + "/test"

        cache = Cache("test", cache_parentname="parent")
        path = self.path + "/parent/test"

        If cache file does not exist it is created
        If cache folder does not exist it is created
        """
        self.path = f"{Path.home()}/.cache/nataili/"
        if cache_parentname:
            self.path = os.path.join(self.path, cache_parentname)
        self.cache_dir = os.path.join(self.path, cache_name)
        if cache_subname:
            self.cache_dir = os.path.join(self.cache_dir, cache_subname)
        self.kv = {}
        self.cache_file = os.path.join(self.cache_dir, "cache.json")
        os.makedirs(self.cache_dir, exist_ok=True)
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                self.kv = json.load(f)
        else:
            self.kv = {}
            json.dump(self.kv, open(self.cache_file, "w"))

    def flush(self):
        """
        Flushes the cache to disk
        """
        json.dump(self.kv, open(self.cache_file, "w"))
