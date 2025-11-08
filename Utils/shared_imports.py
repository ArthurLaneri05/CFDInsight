# Import Global Variables
from global_settings import *

# Remove system Python site-packages to avoid conflicts
import sys
sys.path = [p for p in sys.path if "dist-packages" not in p]

# Typing-related libs
from dataclasses import dataclass
from typing import *
from abc import ABC, abstractmethod

# Other Libs
import os, shutil
from pathlib import Path
import numpy as np

# Import paraview
from paraview.simple import *           # type: ignore

# Add ParaView's Python path explicitly
if paraview_path not in sys.path:
    sys.path.insert(0, paraview_path)  # Insert at beginning for priority

# Import ParaView's additional modules
#from paraview.servermanager import LoadDistributedPlugin    # type: ignore
for module in additional_modules:
    LoadPlugin(module, remote=False, ns=globals()) # type: ignore

# Silence ParaView's warnings
# from vtk import vtkOutputWindow               # type: ignore
# vtkOutputWindow().SetGlobalWarningDisplay(0)  # 0=Off, 1=On

# Get the path to the case folder 
case_folder_path = os.getcwd()

# Add caseFolderPath to Path
sys.path.append(case_folder_path)

# Get the path to the CFDInsight_output folder
CFDInsight_outputPath = os.path.join(case_folder_path, output_folder_title)

# Calculate title characters per line
title_characters_per_line = int((img_size_in[0] * img_dpi) / (title_font_size * alpha))