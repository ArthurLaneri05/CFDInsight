from Utils.shared_imports import *
from Structs.field_options import Field_options, LICField_options
from Structs.case_types import CaseType_Settings, ExternalAerodynamics, TurboMachinery
from Functions.Slices import Slices_Settings, clipping_box
from Functions.Surfaces import Surfaces_Settings
from Functions.IntersectionPlots import IntersectionPlots_Settings

""" Case setup functions """
def domain_setup(CaseType_settings: CaseType_Settings,
                 Function_settings: Union[Surfaces_Settings, 
                                          Slices_Settings, 
                                          IntersectionPlots_Settings]):
    """
    Set up a ParaView visualization pipeline for a CFD case, with optional clipping of the domain,
    symmetry reflection and orientation correction based on initial flow angles (alpha, beta).
    """

    ### Reset the session
    ResetSession()  # type: ignore

    ### Define "clip_domain", "circular_pattern", "reflect", "derotate" flags
    clip_domain = isinstance(Function_settings, Slices_Settings) and Function_settings.clipping_region is not None
    # circular_pattern = isinstance(CaseType_settings, TurboMachinery)
    circular_pattern = False
    reflect = not isinstance(CaseType_settings, TurboMachinery) and CaseType_settings.variant == "Symmetric"
    
    derotate = isinstance(CaseType_settings, ExternalAerodynamics) and CaseType_settings.de_rotate_domain


    ### Load the ghost Meshes (if the function settings expect it)
    meshes_list = []
    if hasattr(Function_settings, "ghost_meshes"):
        for mesh_name in Function_settings.ghost_meshes.names:
            # Path without extension
            base_path = os.path.join(case_folder_path, "constant", "triSurface", mesh_name)

            # Decide whether it's .stl or .obj, then use the appropriate function to import
            current_mesh = None
            if os.path.isfile(base_path + ".stl"):
                path = base_path + ".stl"
                current_mesh = STLReader(registrationName=mesh_name, FileName=path)    # type: ignore
            elif os.path.isfile(base_path + ".obj"):
                path = base_path + ".obj"
                current_mesh = WavefrontOBJReader(registrationName=mesh_name, FileName=path)    # type: ignore
            else:
                raise FileNotFoundError(
                    f"Mesh {mesh_name} not found in {case_folder_path}/constant/triSurface"
                )

            meshes_list.append(current_mesh)

    # Stack the meshes (if present) into a single object
    ghost_meshes = AppendGeometry(Input=meshes_list) if meshes_list else None    # type: ignore        


    ### Load the OpenFOAM simulation case
    foam_file = str(CaseType_settings.foam_file_name).strip()   # remove hidden whitespace/newlines
    case_root = os.path.realpath(case_folder_path)
    foam_path = os.path.abspath(os.path.join(case_root, foam_file))
    if not os.path.isfile(foam_path):
        raise FileNotFoundError(f"The path to the foam file is invalid on the server: {foam_path}")

    domain = OpenFOAMReader(FileName=foam_path)   # type: ignore

    ### Set mesh regions, create point data from cell data
    #   (set after metadata so enums are available; if it fails, fall back to internalMesh later)
    
    # Auto-detect decomposed vs reconstructed
    is_decomposed = os.path.isdir(os.path.join(case_root, "processor0"))
    try:
        domain.CaseType = "Decomposed Case" if is_decomposed else "Reconstructed Case"
    except Exception:
        pass

    # Decompose polyhedra
    try:
        domain.DecomposePolyhedra = 1
    except Exception:
        pass

    # Generate point data from cell data
    try:
        domain.Createcelltopointfiltereddata = 1
    except Exception:
        pass

    # Pull metadata before changing selections; if it fails, the caller can handle/raise
    try:
        UpdatePipelineInformation(proxy=domain)    # type: ignore
    except Exception:
        # Do not crash here; let the caller decide. Return early if desired.
        # raise RuntimeError(f"OpenFOAM reader failed during metadata update for {foam_path}") from e
        pass


    ### Set mesh regions (fallback to internalMesh if needed)
    try:
        domain.MeshRegions = Function_settings.mesh_regions
    except Exception:
        domain.MeshRegions = ["internalMesh"]


    ### Do actions on the source depending on flags
    if circular_pattern:
        symm_domain  = circular_symmetry(domain,       CaseType_settings.axis_of_rotation, CaseType_settings.n_wedges)
        symm_meshes  = circular_symmetry(ghost_meshes, CaseType_settings.axis_of_rotation, CaseType_settings.n_wedges)
        domain       = symm_domain
        ghost_meshes = symm_meshes

    if reflect:
        reflected_domain = mirror_domain(domain,       CaseType_settings.mirroring_axis)
        reflected_meshes = mirror_domain(ghost_meshes, CaseType_settings.mirroring_axis)
        domain       = reflected_domain
        ghost_meshes = reflected_meshes

    if clip_domain:
        clipped_domain = clip_domain_wBox(domain, Function_settings.clipping_region)
        domain = clipped_domain
        
    if derotate:
        # Retrieve flow angles from initial conditions
        alpha = CaseType_settings.alpha
        beta  = CaseType_settings.beta
        angles_deg = [0.0, -alpha, -beta]

        derotated_domain = transform_domain(domain, angles_deg)
        derotated_meshes = transform_domain(ghost_meshes, angles_deg)

        domain       = derotated_domain
        ghost_meshes = derotated_meshes


    ### Add unknown fields
    domain_w_additional_fields = add_custom_fields(domain, all_fields_info=Function_settings.fields)
    domain = domain_w_additional_fields


    ### Return the transformed source
    return domain, ghost_meshes


""" Creation of additional fields """
def add_custom_fields(domain, all_fields_info: List[Union[Field_options, LICField_options]]):
    """
    Set up fields not present in the raw CFD results.
    Note: expressions of unknown fields will have their 'computed_by_calculator'
    flag set to True.
    """

    case_with_fields = domain
    case_with_fields.UpdatePipeline()
    
    # Go through all fields and look at the ones with unknown expressions
    for field_info in all_fields_info:
        if (field_info.expression not in case_with_fields.PointData.keys()
            and field_info.expression is not None):
            
            # Create a calculator object if needed
            case_with_fields = Calculator(Input=case_with_fields)  # type: ignore
            case_with_fields.ResultArrayName = field_info.title
            case_with_fields.Function = field_info.expression
            
            # Set computed_by_calculator flag to True!
            field_info.computed_by_calculator = True

    return case_with_fields


""" Clip domain using a box """
def clip_domain_wBox(domain, clipping_box: clipping_box):
    """
    Clips the domain using a "clipping_box" object.
    """

    # Define corner position
    corner_pos = (clipping_box.center 
                  -0.5 * clipping_box.dimensions).tolist()

    # Create a clip object
    clipped_domain = Clip(Input=domain)   # type: ignore
    clipped_domain.ClipType = 'Box'
    clipped_domain.ClipType.Position = corner_pos
    clipped_domain.ClipType.Rotation = clipping_box.rotation
    clipped_domain.ClipType.Length = clipping_box.dimensions

    return clipped_domain


""" Transform domain """
def transform_domain(domain, 
                  angles_deg: List[float], 
                  translation_m: List[float] = [0,0,0]):
    """
    Note: the translation is applied after the rotation!
    """
    
    transformed_domain = Transform(Input=domain)   # type: ignore
    transformed_domain.TransformAllInputVectors = 0     # Disable automatic vector transformation
    transformed_domain.Transform.Rotate = angles_deg
    transformed_domain.Transform.Translate = translation_m

    return transformed_domain


""" Mirror domain along a plane """
def mirror_domain(domain, mirroring_axis: List[float]):
    mirrored_domain = AxisAlignedReflect(Input=domain)   # type: ignore

    mirrored_domain.Set(
        CopyInput=1,
        ReflectAllInputArrays=1,
    )
    mirrored_domain.ReflectionPlane.Set(
        Origin=[0,0,0],
        Normal=mirroring_axis,
        Offset=0.0,
        AlwaysSnapToNearestAxis=1,
    )

    return mirrored_domain
  
    
""" Circular symmetry along axis """
def circular_symmetry(domain, axis: str, N_wedges: int, N_patches: int):

    # Get block indices:
    block_indices = np.arange(0, N_patches + 1, 1) + 2
    
    periodic_domain = AngularPeriodicFilter(Input=domain)    # type: ignore
    periodic_domain.BlockIndices = block_indices.tolist()
    periodic_domain.IterationMode = 'Maximum'
    periodic_domain.RotationMode = 'Direct Angle'
    periodic_domain.RotationAngle = 360.0 / N_wedges
    periodic_domain.Axis = axis
    periodic_domain.Center = [0.0, 0.0, 0.0]
    periodic_domain.ComputeRotationsOnTheFly = 1

    return periodic_domain
