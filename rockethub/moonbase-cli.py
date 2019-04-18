# -*- coding: utf-8 -*-
"""
    rocketbase.cli
    ~~~~~~~~~

    A simple command line application to set up MoonBase.

    :copyright: © 2019 by the Mirage team.
"""

import os
import sys

from rockethub.rocket import Rocket

import fire

def land(rocket: str, folder_path = 'rockets', chunk_size = 512):
	assert chunk_size>0 , "Please enter a valid chunk size"
	assert len(rocket)>0, "Please enter a valid Rocket name"
	assert len(folder_path)>0, "Please enter a valid folder path"
	Rocket.land(rocket, folder_path, chunk_size)
	return True

def launch(rocket: str, isPrivate=False, folder_path = "rockets"):
	assert len(rocket)>0, "Please enter a valid Rocket name"
	assert len(folder_path)>0, "Please enter a valid folder path"
	return Rocket.launch(rocket, isPrivate, folder_path)

def main():
	fire.Fire(name="moonbase")

if __name__ == '__main__':
	main()