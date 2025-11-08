# Selective import of just the data structures the user can define in "Preferences.py"

### Case Type
from Structs.case_types import (CaseType_Settings,
                                ExternalAerodynamics)

### Fields
from Structs.field_options import (Field_options,
                                   LICField_options,
                                   GhostMeshes)

### Coefficients
from Functions.Coefficients import (Coefficients_Settings,
                                    BodyPatch,
                                    SplitBodyPatch,
                                    RotorPatch)

### Slices
from Functions.Slices import (Slices_Settings, 
                              SlicePlane_Translate_View,
                              SlicePlane_Rotate_View, 
                              SliceCylinder_View, 
                              clipping_box)

### Surfaces
from Functions.Surfaces import (Surfaces_Settings,
                                StandardSurface_View)

### Intersection Plots
from Functions.IntersectionPlots import (IntersectionPlots_Settings,
                                         IntersectionPlane_Plot,
                                         IntersectionCylinder_Plot)

### Delta Surfaces
from Functions.DeltaSurfaces import DeltaSurfaces_Settings

### Generic variable import
from Structs.generic_variable import variable

### Useful libs
from typing import Tuple, Dict
import numpy as np