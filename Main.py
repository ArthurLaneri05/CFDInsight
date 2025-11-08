# CFDInsight©, by Arthur Laneri, 2025.
# License: AGPL-3.0-or-later


# Pvserver set-up utils
from Utils.pvserver_utils import *

# Input data management utils 
from Utils.input_data_manag_utils import *

# Folder opening util
from Utils.os_utils import open_folder

# Function_Settings struct import
from Structs.generic_function import Function_Settings

# Import of all functions
import Functions 

# Handle Keywords, Preferences import from case dict
try:
    import Keywords, Preferences    # type: ignore
except ImportError:
    raise Warning("Either the \"Keywords.py\" or the \"Preferences.py\" module isn't present in the case folder")


"""
===================================================================
SET UP PVSERVER
===================================================================
"""

connection_timeout_s = 15
pv_server_active = activate_local_pvserver(np=pvserver_np,
                                           server_port=pvserver_port,
                                           timeout_s=connection_timeout_s)

if not pv_server_active:
    raise TimeoutError(f"pvserver did not start listening on port {pvserver_port} within {connection_timeout_s}s")


"""
===================================================================
CONNECT TO PVSERVER
===================================================================
"""

pv_server_connected = connect_to_local_pvserver(server_port=pvserver_port)

if not pv_server_connected:
    raise ConnectionError(f"Failed to connect to local pvserver on 127.0.0.1:{pvserver_port}")


"""
===================================================================
GET INPUTS
===================================================================
"""

# Get Input names:
keys = Keywords.get_keywords()

# Get InitialConditions Inputs:
initialConditions_data = dict_from_initialConditions(keys)

# Get Case and Function Settings:
(CaseType_settings, 
Coefficients_settings,
Slices_settings,
Surfaces_settings,
IntersectionPlots_settings,
DeltaSurfaces_settings) =  Preferences.get_settings(initialConditions_data)

# Create caseName and baselineName strings
caseName = get_caseName()
baselineName = get_caseName(DeltaSurfaces_settings.baseline_path)

# Display CaseType and Function Settings:
display_case_information_and_selected_funtions(
    caseName,
    CaseType_settings,
    [Coefficients_settings,
    Slices_settings,
    Surfaces_settings,
    IntersectionPlots_settings,
    DeltaSurfaces_settings])

# Create Model Configuration Summary string to display in Coefficients avgs, Slices and Surfaces images
modelConfig_summary = create_summary(initialConditions_data, 
                                     characters_per_line=title_characters_per_line)

modelConfig_summary_coeffs = create_summary(initialConditions_data, 
                                            characters_per_line = None,
                                            for_coefficients = True)


"""
===================================================================
CREATE OUTPUT FOLDER
===================================================================
"""

# Create the directory CFDInsight_outputPath
os.makedirs(CFDInsight_outputPath, exist_ok=True)
 

"""
===================================================================
ACTIVATE FUNCTIONS
===================================================================
"""

def func_Wrapper(function: Callable, settings: Function_Settings, args):

    if settings.run:
        print(f"Generating {settings.output_name}...")

        # Call the function
        function(settings, *args)

        print("   -> Done!\n")

##### Coefficients
func_Wrapper(function=Functions.Coefficients.Main, 
             settings=Coefficients_settings,
             args=[CaseType_settings,
                   caseName, 
                   modelConfig_summary_coeffs])

##### Slices
func_Wrapper(function=Functions.Slices.Main, 
             settings=Slices_settings,
             args=[CaseType_settings,
                   caseName, 
                   modelConfig_summary])

##### Surfaces
func_Wrapper(function=Functions.Surfaces.Main,
             settings=Surfaces_settings,
             args=[CaseType_settings,
                   caseName, 
                   modelConfig_summary]) 

##### Intersection Plots
func_Wrapper(function=Functions.IntersectionPlots.Main,
             settings=IntersectionPlots_settings,
             args=[CaseType_settings,
                   caseName, 
                   modelConfig_summary]) 

##### DeltaSurfaces
func_Wrapper(function=Functions.DeltaSurfaces.Main,
             settings=DeltaSurfaces_settings,
             args=[Surfaces_settings,
                   caseName, 
                   baselineName]) 
 

"""
===================================================================
OPEN OUTPUT FOLDER
===================================================================
"""
# open_folder(CFDInsight_outputPath)