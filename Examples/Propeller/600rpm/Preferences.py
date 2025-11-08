# CFDInsight©, by Arthur Laneri, 2025.
# License: AGPL-3.0-or-later
#
# Central configuration for post‑processing.
# For full behavior and edge cases, hover on the functions/objects to read their docstrings.


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
    Assemble settings for one CFD case.
    Ranges, normalizations, and conventions are tied to `initialConditions`.
    See module/class/function docstrings for validation rules and side effects.
    """

    # ------------------------------------------------------------------
    # Case Type Settings
    # ------------------------------------------------------------------
    # Problem definition: domain asymmetry/symmetry (with mirroring plane),
    # optional de‑rotation for consistent visualization across different aerodynamic angles (α, β).
    # Note: Vectors and positions in post-processing are defined with respect to the *derotated* geometry.

    # Case naming convention used in titles/outputs:
    # For a case in a dict `.../Propeller/600rpm`, the displayed caseName is `Propeller (600rpm)`.
    # This makes multi‑OP comparisons clearer.
    CaseType_settings = ExternalAerodynamics(
        foam_file_name="Prop.foam",
        variant="Asymmetric",      # choose 'Symmetric' + mirroring_plane for half-domain sims
        mirroring_plane=None,      # e.g. 'XZ'
        de_rotate_domain=False,    # Can be `True` only when alpha, beta values are provided
        alpha=None,                
        beta=None,                 
    )

    # ------------------------------------------------------------------
    # Sync with 'initialConditions'
    # ------------------------------------------------------------------
    U_inf = initialConditions_data["magU"].value        
    rho_inf = initialConditions_data["rho"].value      
    rotor_omega_rpm = initialConditions_data["omega"].value

    # Derived reference
    rotor_D = 0.4
    U_ref_radial = rotor_omega_rpm * np.pi/30.0 * rotor_D/2

    # ------------------------------------------------------------------
    # Coefficients Settings
    # ------------------------------------------------------------------
    # Reference geometry defaults: used unless per‑patch values
    # are specified in Coefficients settings below.
    A_ref_default = 3.14 * rotor_D/4**2
    c_ref_default = rotor_D
    b_ref_default = rotor_D
    CofR_default = [0, 0, 0]

    # Patch definitions
    rotor_patch = RotorPatch(
        name="Propeller",
        origin=[0, 0, 0],
        x_axis=[0, 1, 0],   # local drag direction in rotor frame
        z_axis=[-1, 0, 0],  # local thrust direction
        D=rotor_D,
        omega=rotor_omega_rpm,
    )
    body_patch = BodyPatch(name="Hub")

    # Axes convention note (explicit): when α, β are meaningful, switch to
    # 'body_axes' if you want moments about body axes; default is 'wind_axes' (along standard X, Y, Z axes).
    # Additional entries control how the coefficient values are averaged/presented, and what residuals have to be shown.
    Coefficients_settings = Coefficients_Settings(
        run=True,
        A_ref_default=A_ref_default,
        c_ref_default=c_ref_default,
        b_ref_default=b_ref_default,
        CofR_default=CofR_default,
        rho_inf=rho_inf,
        U_inf_mag=U_inf,
        moment_coefficients_convention='wind_axes',
        coefficients_decimal_digits=3,
        coefficients_avg_iters=20,
        complete_patches=[body_patch],
        split_patches=[],            # use only in symmetric (mirrored) domains
        rotor_patches=[rotor_patch],
        residuals_of_interest=[
            "Ux_final", "Uy_final", "Uz_final",
            "p_final", "omega_final", "k_final",
        ],
    )

    # ------------------------------------------------------------------
    # Fields (used by Slices/Surfaces/IntersectionPlots)
    # ------------------------------------------------------------------

    # The 'Field_options' and 'LICField_options' dataclasses contain all settings regarding fields creation and visualization.
    # IMPORTANT: not all fields can be visualized in every View/Function! See their docstrings for more details.
    # Ranges and expressions can be optionally be expressed in terms of values extracted in initialConditions_data.
    Cp = Field_options(
        title="Cp",
        expression=f"pMean/(0.5*{U_ref_radial}*{U_ref_radial})",   # normalization by U_ref_radial
        field_component="Magnitude",
        field_min=-1.0,
        field_max=1.0,
        color_preset="Fast",
        num_colors=16,
        cbar_tick_stride=2,
        # Export grayscale companion for DeltaSurfaces (see Surfaces module).
        prepare_for_DeltaSurfaces=True,
        delta_field_max=.15,
        # Light grid overlay on slices to aid reading distances.
        background_grid=True
    )

    UMean = Field_options(
        title="UMean",
        expression="UMean",
        field_component="Magnitude",
        field_min=0.0,
        field_max=U_ref_radial,
        color_preset="Jet",
        num_colors=16,
        cbar_tick_stride=2,
        prepare_for_DeltaSurfaces=False,
    )

    UNear = Field_options(
        title="UNear_radial",
        expression="UNear", 
        field_component="Magnitude",
        source='CELLS',
        field_min=0.0,
        field_max=U_ref_radial,
        color_preset="Jet",
        num_colors=16,
        cbar_tick_stride=2,
        prepare_for_DeltaSurfaces=True,
        delta_field_max=2,
    )

    # LIC field. Used only in Slices and Surfaces. 
    # Exhibit tiling due to the way the image is generated in parallelized pipeline in pvserver :(
    LIC = LICField_options(
        title="LIC_Cp",
        expression="Cp",
        field_component="Magnitude",
        field_min=0.0,
        field_max=U_ref_radial,
        color_preset="Jet",
        LIC_input_vectors='UNear',
        LIC_color_mode='Multiply',
        LIC_enhance_contrast='LIC and Color',
        LIC_num_steps=40,
        LIC_step_size=.5
    )

    # Mesh field to showcase 'Surface With Edges' representation type
    Mesh = Field_options(
        title="Mesh",
        expression=None,
        representation_type='Surface With Edges'
    )

    # ------------------------------------------------------------------
    # Slices
    # ------------------------------------------------------------------
    # Views: translate along X, rotate around axis (θ), and cylindrical (unwrapped).
    
    # While it is useless in this context, the `clipping_region` entry is useful 
    # to limit sampling region and accelerate slicing operations when only a domain subset is of interest.
    
    X_offsets = np.round(np.arange(-0.1, 0.302, 0.02), 3)
    angles_deg = np.linspace(-36, 36, 21)[:-1]
    cyl_radii = np.array([0.05, .1, .15])

    Slices_settings = Slices_Settings(
        run=True,
        mesh_regions=["internalMesh"],
        ghost_meshes=GhostMeshes(names=[], color=[0, 0, 0], opacity=0.0),
        views=[
            SlicePlane_Translate_View(
                title="X",
                plane_normal=[1.0, 0.0, 0.0],
                focus_position=[0.0, 0.0, 0.0],
                viewUp_direction=[0.0, 0.0, 1.0],
                parallel_scale=0.3,
                offsets_array=X_offsets,
            ),
            SlicePlane_Rotate_View(
                title="Theta",
                focus_position=[0.0, 0.0, 0.0],
                x_axis=[0, 1, 0],
                z_axis=[-1, 0, 0],
                viewUp_direction="z_axis",
                parallel_scale=0.3,
                angles_array=angles_deg,
            ),
            SliceCylinder_View(
                title="R",
                focus_position=[0.0, 0.0, 0.0],
                x_axis=[0, 1, 0],
                z_axis=[-1, 0, 0],
                height=0.6,
                mesh_element_size=1e-3,
                parallel_scale=0.6,
                radii_array=cyl_radii,
            ),
        ],
        fields=[Cp, UMean, LIC],
        clipping_region=clipping_box(
            center=[0, 0, 0],
            rotation=[0, 0, 0],
            dimensions=[1, 1, 1]
        ),
    )

    # ------------------------------------------------------------------
    # Surfaces
    # ------------------------------------------------------------------
    # Surface rendering on specified mesh regions. Ghost meshes add context
    # (e.g., semi‑transparent MRF). Fields with prepare_for_DeltaSurfaces=True
    # are cloned to grayscale and exported for the DeltaSurfaces workflow.
    Surfaces_settings = Surfaces_Settings(
        run=True,
        mesh_regions=["group/Propeller_Group"],
        ghost_meshes=GhostMeshes(names=["MRF"], color=[1, 1, 1], opacity=0.1),
        views=[
            StandardSurface_View(
                title="Front",
                camera_position=[-1.0, 0.0, 0.0],
                focus_position=[0.0, 0.0, 0.0],
                viewUp_direction=[0.0, 0.0, 1.0],
                interaction_mode="2D",
                parallel_scale=0.35,
            ),
            StandardSurface_View(
                title="ISO",
                camera_position=[-0.5, -0.8, 0.8],
                focus_position=[0.0, 0.0, 0.0],
                viewUp_direction=[0.0, 0.0, 1.0],
                interaction_mode="3D",
            ),
        ],
        fields=[Cp, UNear, LIC, Mesh],
    )

    # ------------------------------------------------------------------
    # Intersection Plots
    # ------------------------------------------------------------------
    # Geometric intersections with stacked fields per plot. Each tuple in
    # `field_tuples` becomes a vertically stacked set of panels with a common
    # abscissa. Cylindrical intersections use unwrapped arc‑length.
    # The settings and views are similar to 'Slices'
    IntersectionPlots_settings = IntersectionPlots_Settings(
        run=True,
        mesh_regions=["group/Propeller_Group"],
        plots=[
            IntersectionCylinder_Plot(
                title="Cylindrical_plot",
                center=[0.0, 0.0, 0.0],
                x_axis=[0, 1, 0],
                z_axis=[1, 0, 0],
                offsets_array=cyl_radii,
            ),
            IntersectionPlane_Plot(
                title="Planar_plot",
                plane_normal=[0, 1, 0],
                graph_x_axis=[1, 0, 0],
                offsets_array=cyl_radii
            )
        ],
        field_tuples=[(Cp, UNear)],
    )

    # ------------------------------------------------------------------
    # Delta Surfaces
    # ------------------------------------------------------------------
    # Two‑case workflow:
    #  1) Run 'Surfaces' in this case:
    #     Note: only the fields with `prepare_for_DeltaSurfaces=True` will be displayed,
    #           and the delta-visualization will show values in the range (-delta_field_max, delta_field_max).
    #  2) Create `baseline` case:
    #     - Assuming the path to the current case is "path/to/Propeller/600rpm", 
    #       copy the contents in "path/to/Propeller/1200rpm". 
    #     - In the `initialConditions` file of the cloned case, set `omega` to 1200.
    #     - Delete the `150` folder in the case (so that you keep the mesh, but ditch the CFD results).
    #     - Re-run the CFD with the changed BC
    #     - Run just the `Surfaces` by opening a new terminal there, without changing Views & other settings.
    #  3) Run and look at the results!
    #     - Go back to this case, put the path to the baseline case in `baseline_path` 
    #       (has to be absolute) below and run `DeltaSurfaces`.

    DeltaSurfaces_settings = DeltaSurfaces_Settings(
        run=False,
        baseline_path="~/absolute/path/to/Propeller/1200rpm",
    )

    # ------------------------------------------------------------------
    # Return (order matters)
    # ------------------------------------------------------------------
    return (
        CaseType_settings,
        Coefficients_settings,
        Slices_settings,
        Surfaces_settings,
        IntersectionPlots_settings,
        DeltaSurfaces_settings,
    )
