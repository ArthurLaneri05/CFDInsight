# CFDInsight©, by Arthur Laneri, 2025.
# License: AGPL-3.0-or-later


### Imports
from Utils.user_imports import *


def get_settings(
        initialConditions_data: Dict[str, variable]
        ) -> Tuple[CaseType_Settings,
                   Coefficients_Settings,
                   Slices_Settings,
                   Surfaces_Settings,
                   DeltaSurfaces_Settings]:
    """
    Assemble and return all post-processing settings as a tuple.

    This function is the *central orchestrator* for a single CFD case:
    - it uses the CFD inputs coming from `initialConditions` via `Keywords.py`
      to adapt the behaviour of the post-processing pipeline.
    - it returns a single, ordered bundle of settings that the main pipeline
      can just iterate over.
    """

    # ----------------------------
    # Case Type Settings
    # ----------------------------
    CaseType_settings = ExternalAerodynamics(
        foam_file_name=,
        variant=,
        mirroring_plane=,
        de_rotate_domain=,
        alpha=,
        beta=,
    )

    # ----------------------------
    # Sync with 'initialConditions'
    # ----------------------------
    variable1 = initialConditions_data['key'].value

    # ----------------------------
    # Coefficients Settings
    # ----------------------------
    rotor_patch = RotorPatch(
        name=,
        origin=,
        x_axis=,
        z_axis=,
        D=,
        omega=,
    )

    body_patch = BodyPatch(
        name=,
        A_ref=,
	c_ref=,
	b_ref=,
	CofR=,
        )
        
    split_body_patch = BodyPatch(
        name=,
        A_ref=,
	c_ref=,
	b_ref=,
	CofR=,
        )

    Coefficients_settings = Coefficients_Settings(
        run=False,
        A_ref_default=,
        c_ref_default=,
        b_ref_default=,
        CofR_default=,
        rho_inf=,
        U_inf_mag=,
        moment_coefficients_convention=,
        coefficients_decimal_digits=,
        coefficients_avg_iters=,
        complete_patches=[body_patch],
        split_patches=[split_body_patch],
        rotor_patches=[rotor_patch],
        residuals_of_interest=,
    )

    # ----------------------------
    # Fields Setup
    # ----------------------------
    field = Field_options(
        title=,
        expression=,   
        field_component=,
        source=,
        representation_type=,
        field_min=,
        field_max=,
        color_preset=,
        num_colors=,
        cbar_tick_stride=,
        prepare_for_DeltaSurfaces=,
        delta_field_max=,
        background_grid=        # Explain
    )

    LIC_field = LICField_options(
        title=,
        expression=,   
        field_component=,
        source=,
        representation_type=,
        field_min=,
        field_max=,
        color_preset=,
        num_colors=,
        cbar_tick_stride=,
        
        LIC_input_vectors=,
        LIC_color_mode=,
        LIC_enhance_contrast=,
        LIC_num_steps=,
        LIC_step_size=
    )

    # ----------------------------
    # Slices Settings
    # ----------------------------
    Slices_settings = Slices_Settings(
        run=False,
        mesh_regions=,      
        ghost_meshes=GhostMeshes(names=, color=, opacity=),
        views=[
            SlicePlane_Translate_View(
                title=,
                plane_normal=,
                focus_position=,
                viewUp_direction=,
                parallel_scale=,
                offsets_array=,
            ),
            SlicePlane_Rotate_View(     
                title=,
                focus_position=,
                x_axis=,
                z_axis=,
                viewUp_direction=,
                parallel_scale=,
                angles_array=,
            ),
            SliceCylinder_View(    
                title=,
                focus_position=,
                x_axis=,
                z_axis=,
                height=,
                mesh_element_size=,
                parallel_scale=,
                radii_array=,
            ),
        ],
        fields=,
        clipping_region=clipping_box(center=, rotation=, dimensions=),
    )

    # ----------------------------
    # Surfaces Settings
    # ----------------------------
    # explain the function of ghost meshes
    Surfaces_settings = Surfaces_Settings(
        run=False,
        mesh_regions=,
        ghost_meshes=GhostMeshes(names=, color=, opacity=),
        views=[
            StandardSurface_View(
                title=,
                camera_position=,
                focus_position=,
                viewUp_direction=,
                interaction_mode=,
                parallel_scale=,
            ),
        ],
        fields=,
    )

    # ----------------------------
    # Intersection Plots
    # ----------------------------
    IntersectionPlots_settings = IntersectionPlots_Settings(
        run=False,
        mesh_regions=,
        plots=[
            IntersectionCylinder_Plot(
                title=,
                center=,
                x_axis=,
                z_axis=,
                offsets_array=,
            ),
            IntersectionPlane_Plot(
                title=,
                plane_normal=,
                graph_x_axis=,
                offsets_array=,
            )
        ],
        field_tuples=,
    )

    # ----------------------------
    # Delta Surfaces
    # ----------------------------
    DeltaSurfaces_settings = DeltaSurfaces_Settings(
        run=False,
        baseline_path=,
    )

    # ----------------------------
    # Return
    # ----------------------------
    return (
        CaseType_settings,
        Coefficients_settings,
        Slices_settings,
        Surfaces_settings,
        IntersectionPlots_settings,
        DeltaSurfaces_settings,
    )

