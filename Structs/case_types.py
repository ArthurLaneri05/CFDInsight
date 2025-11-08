from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Tuple


@dataclass
class CaseType_Settings(ABC):
    """
    Base class for all simulation case types.
    
    Provides a common interface for different types of computational fluid dynamics
    simulations with shared functionality and type identification.
    """
    
    foam_file_name: str
    """Name of the file opened by ParaView"""

    def __set_type__(self):
        """String identifier for the specific case type (e.g., 'ExternalAerodynamics', 'TurboMachinery')"""
        type: Tuple[str] = None

    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass
class ExternalAerodynamics(CaseType_Settings):
    """
    Configure simulations of external flow around bodies (aircraft, vehicles,
    buildings) with support for symmetric (half-domain) or asymmetric (full-domain)
    analyses, and optional domain de-rotation to standardize views across AoA/β.
    
    Attributes:
        foam_file_name (str): Name of the file opened by ParaView.

        variant (Literal["Symmetric", "Asymmetric"]): Symmetry treatment of the domain.
            - "Symmetric": exploit geometric symmetry across a coordinate plane to reduce the domain.
            - "Asymmetric": full 3D analysis with no symmetry assumptions.

        mirroring_plane (Optional[Literal["XY", "YZ", "XZ"]]): Coordinate plane used for symmetry mirroring
            when `variant == "Symmetric"`. Options:
            - "XY": mirror about the XY-plane (normal +Z)
            - "YZ": mirror about the YZ-plane (normal +X)
            - "XZ": mirror about the XZ-plane (normal +Y)
            Must be None when `variant == "Asymmetric"`.

        de_rotate_domain (bool): Whether to apply de-rotation so that Views remain identical across cases
            at different flow angles. When True, expects valid `alpha` (AoA) and `beta` (sideslip), both in **degrees**.
                
            *Note:* `Vectors` and `positions` in post-processing are defined with respect to the **derotated** geometry.

        alpha (float): Angle of attack in degrees. Used only when `de_rotate_domain` is True.

        beta (float): Sideslip angle in degrees. Used only when `de_rotate_domain` is True.
            Must be 0.0 if `variant == "Symmetric"`.

        type (Tuple[str]): Derived. Automatically set to ("ExternalAerodynamics",).

        mirroring_axis (Optional[List[int]]): Derived. Unit axis normal corresponding to `mirroring_plane`,
            mapped as {"XY": [0, 0, 1], "YZ": [1, 0, 0], "XZ": [0, 1, 0]}; None if mirroring is not used.
"""

    variant: Literal["Symmetric", "Asymmetric"]
    mirroring_plane: Optional[Literal["XY", "YZ", "XZ"]]
    de_rotate_domain: bool
    alpha: float
    beta: float

    def __post_init__(self):
        # Set the type automatically as an immutable tuple
        self.type = ("ExternalAerodynamics",)

        # Validate mirroring_plane condition
        if self.variant == "Symmetric" and self.mirroring_plane is None:
            raise ValueError("mirroring_plane must be specified when variant is 'Symmetric'")
        if self.variant == "Asymmetric" and self.mirroring_plane is not None:
            raise ValueError("mirroring_plane can only be specified when variant is 'Symmetric'")

        # Validate beta (must be 0 if variant = "Symmetric")
        if self.variant == "Symmetric" and self.beta != 0.0:
            raise ValueError("beta must be 0.0 if the variant is 'Symmetric'")

        # Set the mirroring axis (if needed)
        self.mirroring_axis = None
        decoding_dict = {"XY": [0, 0, 1], "YZ": [1, 0, 0], "XZ": [0, 1, 0]}
        if self.mirroring_plane:
            self.mirroring_axis = decoding_dict[self.mirroring_plane]

    def __str__(self):
        # Start with case type (first character of type string)
        string = f"CaseType: {self.type[0]}"

        # Add information about the foam file loaded
        string += f"\n\t  Domain loaded from: \"{self.foam_file_name}\""

        # Add variant of the simulation case
        string += f"\n\t  Variant: {self.variant}"

        # Add mirroring plane info if present
        if self.mirroring_plane:
            string += f"\n\t  Mirroring Plane: {self.mirroring_plane}"

        # Add domain de-rotation flag
        string += f"\n\t  Domain de-rotation: {self.de_rotate_domain}"

        # If found, add rotation angles
        if self.alpha and self.beta:
            string += f" [Alpha: {self.alpha:.2f}°; Beta: {self.beta:.2f}°]"

        return string
    
### TurboMachinery placeholder
@dataclass
class TurboMachinery(CaseType_Settings):
    pass


# @dataclass
# class TurboMachinery(CaseType_Settings):
#     """
#     Configuration for turbomachinery simulations.
    
#     Represents simulations involving rotating machinery such as:
#     - Turbines and compressors
#     - Pumps and fans
#     - Propellers and rotors
    
#     Supports periodic wedge analysis for rotational symmetry.
#     """
    
#     axis_of_rotation: List[float]
#     """
#     3D vector defining the axis of rotation [x, y, z].
    
#     Defines the direction vector around which the turbomachinery rotates.
#     Typically normalized to unit length.
#     """
    
#     n_wedges: int
#     """
#     Number of periodic wedges for rotational symmetry analysis.
    
#     Represents the number of identical sectors in the full 360-degree geometry.
#     Must be a positive integer (typically 2, 3, 4, etc. for symmetric blades).
#     """

#     def __post_init__(self):
#         """
#         Initialize and validate turbomachinery case configuration.
        
#         Raises:
#             ValueError: If n_wedges is not a positive integer
#             ValueError: If axis_of_rotation is not a 3D vector
#         """
#         # Set the type automatically as an immutable tuple
#         self.type = ("TurboMachinery",)
        
#         # Validate n_wedges
#         if self.n_wedges < 1:
#             raise ValueError("n_wedges must be > 0")
        
#         # Validate axis_of_rotation
#         if len(self.axis_of_rotation) != 3:
#             raise ValueError("axis_of_rotation must be a 3D vector [x, y, z]")

#     def __str__(self):
#         # Case type
#         string = f"CaseType: {self.type[0]}"

#         # Add information about the foam file loaded
#         string += f"\n\t  Domain loaded from: \"{self.foam_file_name}\""

#         # Add axis of rotation
#         string += f"\n\t  Axis of rotation: [{self.axis_of_rotation[0]:.2}," +\
#                   f" {self.axis_of_rotation[1]:.2}, {self.axis_of_rotation[2]:.2}]"

#         # Add number of wedges
#         string += f"\n\t  Number of wedges: {self.n_wedges}"
        
#         return string

# @dataclass
# class InternalFlow(CaseType):
#     """
#     Configuration for internal flow simulations.
    
#     Represents simulations involving flow through enclosed passages such as:
#     - Ducts and pipes
#     - Heat exchangers
#     - Internal cooling passages
#     - Cardiovascular flows
    
#     Supports both symmetric and asymmetric analysis with optional mirroring.
#     """
    
#     variant: Literal["Symmetric", "Asymmetric"]
#     """
#     Analysis variant determining symmetry treatment:
#     - 'Symmetric': Exploit geometric symmetry to reduce computational domain
#     - 'Asymmetric': Full analysis without symmetry assumptions
#     """
    
#     mirroring_plane: Optional[Literal["XY", "YZ", "XZ"]]
#     """
#     Coordinate plane used for symmetry mirroring (required for Symmetric variant).
    
#     Options:
#     - 'XY': Mirror about XY-plane (z-symmetry)
#     - 'YZ': Mirror about YZ-plane (x-symmetry)
#     - 'XZ': Mirror about XZ-plane (y-symmetry)
    
#     Only applicable when variant is 'Symmetric'.
#     """

#     def __post_init__(self):
#         """
#         Initialize and validate internal flow case configuration.
        
#         Raises:
#             ValueError: If mirroring_plane configuration is inconsistent with variant
#         """
#         # Set the type automatically
#         self.type = "InternalFlow"
        
#         # Validate mirroring_plane condition
#         if self.variant == "Symmetric" and self.mirroring_plane is None:
#             raise ValueError("mirroring_plane must be specified when variant is 'Symmetric'")
#         if self.variant != "Symmetric" and self.mirroring_plane is not None:
#             raise ValueError("mirroring_plane can only be specified when variant is 'Symmetric'")
        
        