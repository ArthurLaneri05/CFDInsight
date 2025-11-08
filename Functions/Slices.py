from Utils.shared_imports import *
from Structs.generic_function import Function_Settings
from Structs.generic_view import View
import matplotlib.pyplot as plt
from string import Template
from Structs.field_options import *
from Structs.case_types import CaseType_Settings
from Utils.vector_manipulation_utils import gram_schmidt_ortogonalization, euler_angles_from_axes_deg
from copy import deepcopy

""" Settings for Preferences.py file """
@dataclass
class Slices_Settings(Function_Settings):
    """
    Settings for surface-based visualization and analysis.
    
    Attributes:
        run (bool): Flag that determines wether the function has to be run.
        mesh_regions (List[str]): List of mesh region names to include in surfaces.
        ghost_meshes (GhostMeshes): Structure that stores options regarding meshes to display along the result.
        views (List[SlicePlane_View, SliceCylinder_View]): List of Surface_View configurations.
        fields (List[Union[Field_options, LICField_options]]): List of field options for visualization.
        clipping_region (Optional[clipping_box]): Optional clipping box to speed-up slicing operations.
    """
    mesh_regions: List[str]
    ghost_meshes: GhostMeshes
    views: List[Union["SlicePlane_Translate_View", 
                      "SlicePlane_Rotate_View",
                      "SliceCylinder_View"]]
    fields: List[Union[Field_options, LICField_options]]
    clipping_region: Optional["clipping_box"] = None

    def __post_init__(self):
        """Initialize with default name for slice settings."""
        self.output_name = "Slices"
        

@dataclass
class clipping_box:
    """
    Defines a 3D clipping box for limiting visualization regions.
    
    Attributes:
        center ([float, float, float]): 3D coordinates of the clipping box center in meters.
        rotation ([float, float, float]): Rotation angles [rx, ry, rz] in degrees.
        dimensions ([float, float, float]): Box dimensions [dx, dy, dz] in meters.
    """
    center: List[float]
    rotation: List[float]
    dimensions: List[float]

    def __post_init__(self):
        self.center     = np.array(self.center)
        self.rotation   = np.array(self.rotation)
        self.dimensions = np.array(self.dimensions)


""" Generic _Slice_View class declaration """
class _Slice_View(View):
    """Abstract base for slice-plane views.

        Provides common camera/setup, field application, image export, and sweep
        logic. Subclasses implement how the slice plane is updated between frames
        (e.g., translating along the normal vs. rotating about an axis).
        """
    
    def __init__(self, 
                 title: str,
                 camera_position: List[str],
                 focus_position: List[float],
                 viewUp_direction: List[float],
                 parallel_scale: float,
                 offsets_array: np.ndarray): 

        # Attributes initialization
        super().__init__(title = title,
                         camera_position = camera_position,
                         focus_position = focus_position,
                         viewUp_direction = viewUp_direction,
                         parallel_scale=parallel_scale,
                         interaction_mode='2D')
        
        # Offsets arrays initialization
        self.offsets_array = offsets_array
        self._current_slice_offset = 0.0     # Placeholder

        # Template string for offset label
        self._offset_label_str: Template = None


    ### Source-related (in addition to super()._create_source())
    @abstractmethod
    def _update_source(self) -> None:
        pass


    ### Image-related
    def _additional_img_ref(self, 
                            field_options: Union[LICField_options, Field_options],
                            fig: plt.Figure,
                            ax: plt.Axes,
                            img: np.ndarray) -> bool:
        """
        Apply function-specific refinements to the image (annotations, overlays, etc.).

        Args:
            field_options (LICField_options, Field_options): Field options.
            fig (matplotlib.Figure): The plot figure.
            ax (matplotlib.axes.Axes): The axes object on which the image is drawn.
            img (numpy.ndarray): The np.array derived from the image.

        Returns:
            bool: Flag used to optionally stop the refinement process. 
                If 'True' the refinement proceeds.
        """
        
        # Add offset label
        ax.annotate(xy=[0, 1], 
                    xycoords='axes fraction',
                    textcoords='offset pixels',
                    xytext=[offset_label_spacing_in*img_dpi,
                            -offset_label_spacing_in*img_dpi],
                    text=self._offset_label_str.substitute(value=f"{self._current_slice_offset:.3f}"), 
                    fontsize=title_font_size,
                    ha='left', va='top'
                    )
        
        # Add grid (Optional)
        if hasattr(field_options, "background_grid") and field_options.background_grid:
            # Show axes
            ax.axis('on')

            # Remove spines
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Remove tick labels and marks
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(left=False, bottom=False, top=False, right=False)

            # Set custom ticks
            N_ticks_per_side = 20
            x_ticks = np.linspace(0, img.shape[1], N_ticks_per_side + 1)[1:-1]
            y_ticks = np.arange(0, img.shape[0], x_ticks[1]- x_ticks[0])[1:]

            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)

            # Show grid
            ax.grid(True, linewidth = .5, alpha = .5)

        return True


    ### Small deviation from super().create_all_images()
    def create_all_images(self) -> None:
        """
        Creates and saves all images related to the view
        """    

        # Cicle through all offset_values
        for offset in self.offsets_array:
            
            # Update "self.current_slice_offset" and slice obj itself
            self._current_slice_offset = offset
            self._update_source()

            # Cicle through all fields inside "self.field_list" and create images
            for field_properties in self.field_list:
                # Apply the field to the model inside the RenderView
                self._apply_field(field_properties)

                # Build img output path
                img_title = f"{self._screenshot_counter}_{self._current_slice_offset:.3f}m.png"
                complete_img_path = os.path.join(output_path, 
                                                 self.title, 
                                                 self._current_field.title, 
                                                 img_title)

                # Export image
                self._export_image(complete_img_path=complete_img_path,
                                   field_options=field_properties)
                
            # Update the screenshot counter to order the images
            self._screenshot_counter += 1


""" SlicePlane_..._View classes declaration """
class SlicePlane_Translate_View(_Slice_View):
    def __init__(self, 
                 title: str,
                 plane_normal: List[float],
                 focus_position: List[float],
                 viewUp_direction: List[float],
                 parallel_scale: float,
                 offsets_array: np.ndarray):

        """
        Fixed-normal slicing view.

        The slice plane keeps a constant normal vector and is translated along that
        normal by the values in `offsets_array` (meters).

        Attributes:
            title (str): Descriptive title for the slice view.
            plane_normal ([float, float, float]): Vector defining the normal direction of the slice plane.
                + The camera is placed at **`camera_distAlongNormal * plane_normal + focus_position`** 
                + `camera_distAlongNormal` is editable in **`CFDInsight/global_settings.py`**.
            focus_position ([float, float, float]): Vector defining the camera focus point.
            viewUp_direction ([float, float, float]): Vector defining the 'up' direction for the camera.
            parallel_scale (float): Scaling factor for parallel projection.
            offsets_array (np.ndarray): Array of plane offset values in **meters**.
        """

        # Attributes initialization
        self.normal_vector = plane_normal
        camera_position = (camera_distAlongNormal * np.array(self.normal_vector) + 
                           np.array(focus_position)).tolist()

        super().__init__(title=title,
                         camera_position=camera_position,
                         focus_position=focus_position,
                         viewUp_direction=viewUp_direction,
                         parallel_scale=parallel_scale,
                         offsets_array=offsets_array)
        
        # Template string for offset label
        self._offset_label_str: Template = Template(r"$$\bf{Offset:}$$ ${value}m")


    ### Source-related
    def _create_source(self, domain):
        """
        Slice the input returning the slice object.
        """
        slice_obj = Slice(Input=domain)    # type: ignore
        slice_obj.SliceType = 'Plane'
        slice_obj.SliceType.Origin = [0, 0, 0]
        slice_obj.SliceType.Normal = self.normal_vector
        slice_obj.SliceOffsetValues = [0.0]     # Placeholder value
        
        return slice_obj
    

    def _update_source(self):
        self.source.SliceOffsetValues = [self._current_slice_offset]


class SlicePlane_Rotate_View(_Slice_View):
    def __init__(self, 
                 title: str,
                 focus_position: List[float],
                 x_axis: List[float],
                 z_axis: List[float],
                 viewUp_direction: Literal['x_axis', 'z_axis'],
                 parallel_scale: float,
                 angles_array: np.ndarray):

        """
        Axis-hinged rotational slicing view.

        The slice plane passes through a fixed pivot point and rotates about a given
        axis by the angles in `angles_array` (degrees).

        Attributes:
            title (str): Descriptive title for the slice view.
            focus_position ([float, float, float]): Vector defining the camera focus point.
            x_axis (list[float, float, float]): Reference direction vector from which the angles are computed. 
                + *A plane with `angle = 0°` will be **parallel** to both **x_axis** and **z_axis**.*
                + *The x_axis vector is automatically orthogonalized with z_axis using 'Gram-Schmidt' orthogonalization.*
            z_axis ([float, float, float]): Vector defining the plane rotation axis.
            viewUp_direction ([float, float, float]): Vector defining the 'up' direction for the camera.
            parallel_scale (float): Scaling factor for parallel projection.
            angles_array (np.ndarray): Array of plane angle values in **degrees**.
        """

        # Create temporary camera position
        camera_position = (camera_distAlongNormal * np.array([1,0,0]) + 
                           np.array(focus_position)).tolist()
        
        # Create viewUp_direction vector
        self.viewUp_key: str = viewUp_direction
        
        # Create ortonormal reference frame
        (self.x_axis, 
         self.y_axis, 
         self.z_axis) = gram_schmidt_ortogonalization(x_axis, z_axis)
        
        super().__init__(title=title,
                         camera_position=camera_position,   # Placeholder value
                         focus_position=focus_position,
                         viewUp_direction= [0,0,1],         # Placeholder
                         parallel_scale=parallel_scale,
                         offsets_array=angles_array)
        
        # Template string for offset label
        self._offset_label_str: Template = Template(r"$$\bf{Angle:}$$ ${value}°")

    ### Source-related
    def _create_source(self, domain):
        """
        Slice the input returning the slice object.
        """
        slice_obj = Slice(Input=domain)    # type: ignore
        slice_obj.SliceType = 'Plane'
        slice_obj.SliceType.Origin = self.focus_position
        slice_obj.SliceType.Normal = [1,0,0]    # Placeholder value
        slice_obj.SliceOffsetValues = [0.0]    
        
        return slice_obj
    

    def _update_source(self):

        # Calculate plane normal (self.y_axis rotated by angle_rad)
        angle_rad = np.deg2rad(self._current_slice_offset)

        plane_normal:np.ndarray = (-self.x_axis * np.sin(angle_rad) + 
                                   self.y_axis * np.cos(angle_rad))
      
        # Update the plane normal
        self.source.SliceType.Normal = plane_normal.tolist()

        # Update the view position
        self.renderView.CameraPosition = (plane_normal * camera_distAlongNormal).tolist()

        # Update the viewUp vector
        x_axis:np.ndarray = (self.x_axis * np.cos(angle_rad) + 
                             self.y_axis * np.sin(angle_rad))
        viewUp_vectors_dict = {'x_axis' : x_axis, 'z_axis' : self.z_axis}   

        # Choose the vector with viewUp_key
        viewUp_dir_vector = viewUp_vectors_dict[self.viewUp_key]

        self.renderView.CameraViewUp = viewUp_dir_vector


""" SliceCylinder_View class declaration """
class SliceCylinder_View(_Slice_View):
    def __init__(self, 
                 title: str,
                 focus_position: List[float],
                 x_axis: List[float],
                 z_axis: List[float],
                 height: float,
                 mesh_element_size: float,
                 parallel_scale: float,
                 radii_array: np.ndarray):
        
        """
        Defines a developed (unwrapped) cylindrical slice view configuration for visualization.

        **Important:** 
        * `Ghost meshes` are not visualized in this view type.
        * For all fields the entry *'representation_type'* will be set to `'Surface'`, 
            as other representation types wouldn't display the edges/points of the domain mesh,
            but of the utility cylindrical mesh used to generate the view instead.
        * If jagged edges show up at the intersections with the edges of the domain, 
            please lower `mesh_element_size` or use a finer domain mesh.

        Attributes:
            title (str): Descriptive title for the slice view.
            focus_position ([float, float, float]): Vector defining the cylinder's center position.
            x_axis (list[float, float, float]): Direction vector of the cylinder’s local x-axis. 
                + *The surface region aligned with **`+x_axis`** is mapped to the **centerline** of the **unwrapped (developed) slice**.*
                + *The x_axis vector is automatically orthogonalized with z_axis using 'Gram-Schmidt' orthogonalization.*
            z_axis ([float, float, float]): Vector defining the cylinder's axis.
            height (float): Cylinder height (> 0).
            mesh_element_size (float): Size of the quad elements of the cylindrical mesh in meters. 
                *`(A value < 1e-3 is advised)`*
            parallel_scale (float): Scaling factor for parallel projection.
            radii_array (np.ndarray): Array of cylinder radii values in **meters** *`(negative values NOT allowed!)`*.
        """

        # Check that no value <0 is present in radii_array
        if np.any(radii_array <= 0.0):
            raise ValueError("radii_array must be strictly > 0 m")
        
        # Check that height is > 0
        if height <= 0:
            raise ValueError("The cylinder height must be > 0")
        
        # Check that radial_mes_resolution is > 0
        if mesh_element_size <= 0:
            raise ValueError("The mesh element size must be > 0")
        

        # Attributes initialization
        self.cylinder_center = focus_position
        self.cylinder_height = height
        self.mesh_element_size = mesh_element_size
        self.vtp_source = None
        self.vtp_source_name = "vtp_cylinder_source.vtp"

        # Create ortonormal reference frame
        (self.x_axis, 
         self.y_axis, 
         self.z_axis) = gram_schmidt_ortogonalization(x_axis, z_axis)
        
        
        # Parent class initialization
        super().__init__(title=title,
                         camera_position=[-1,0,0],
                         focus_position=[0,0,0],
                         viewUp_direction=[0,0,1],
                         parallel_scale=parallel_scale,
                         offsets_array=radii_array)

        # Template string for offset label
        self._offset_label_str: Template = Template(r"$$\bf{Radius:}$$ ${value}m")


    # View building (No ghost meshes!)
    def build(self, 
              domain,
              ghost_meshes,
              caseName: str, 
              modelConfig_summary: str,
              fields_properties: List[Field_options],
              ghost_meshes_properties: GhostMeshes
              ) -> None:
        
        fields_properties = deepcopy(fields_properties)     # Deepcopying to avoid interactions between views

        # Set ghost_meshes as None, set all repr types to 'Surface'
        for i in range(len(fields_properties)):
            fields_properties[i].representation_type = 'Surface'

        super().build(domain=domain,
                      ghost_meshes=None,
                      caseName=caseName,
                      modelConfig_summary=modelConfig_summary,
                      fields_properties=fields_properties,
                      ghost_meshes_properties=ghost_meshes_properties)


    ### Source-related
    def _create_source(self, domain):
        """
        Slice the input returning the slice object.
        """

        import Utils.domain_setup as domain_setup

        # Save a vtp cylinder mesh in the slice folder
        self._current_slice_offset = 1.0     # Placeholder value
        self.save_vtp_cylinder()    # Placeholder object

        # Open the cylinder inside ParaView
        self.vtp_source = XMLPolyDataReader(    # type: ignore  
            FileName=[os.path.join(output_path, self.title, self.vtp_source_name)])
        self.vtp_source.CellArrayStatus = []
        self.vtp_source.PointArrayStatus = []

        # Triangulate the surface
        vtp_triangulated = Triangulate(Input=self.vtp_source)    # type: ignore
        
        # Create a 'Resample With Dataset' to get the slice
        resampled_dataset = ResampleWithDataset(SourceDataArrays=domain,    # type: ignore
                                              DestinationMesh=vtp_triangulated)
        # resampled_dataset.CategoricalData = 0
        # resampled_dataset.PassCellArrays = 1
        # resampled_dataset.PassPointArrays = 1
        # resampled_dataset.PassFieldArrays = 1
        # resampled_dataset.ComputeTolerance = 1
        # resampled_dataset.Tolerance = 2e-16
        # resampled_dataset.MarkBlankPointsAndCells = 1
        # resampled_dataset.CellLocator = 'Static Cell Locator'

        # Keep continuous interpolation
        resampled_dataset.CategoricalData = 0

        # Be explicit about tolerance
        resampled_dataset.ComputeTolerance = 0
        resampled_dataset.Tolerance = 1e-16
        resampled_dataset.MarkBlankPointsAndCells = 1

        
        # Transform the cylinder so that its center is at the origin, 
        # its x_axis is [1,0,0] and its z_axis is [0,0,1]
        angles = euler_angles_from_axes_deg(self.x_axis, 
                                            self.y_axis, 
                                            self.z_axis)


        # Translate 
        translated_dataset = domain_setup.transform_domain(resampled_dataset,
                                                        angles_deg=[0,0,0],
                                                        translation_m=[-self.cylinder_center[0],
                                                                       -self.cylinder_center[1],
                                                                       -self.cylinder_center[2]]
                                                        )
        
        # Rotate
        rotated_dataset = domain_setup.transform_domain(translated_dataset,
                                                        angles_deg=angles)

        # Develop the cylinder surface on the YZ plane
        flattened_dataset = Calculator(Input=rotated_dataset)  # type: ignore
        flattened_dataset.CoordinateResults = 1
        flattened_dataset.Function = f"""coordsZ*kHat + 
        (atan(coordsX/coordsY)/1.570796326 + (coordsY/abs(coordsY))) * {self._current_slice_offset}/.9 * jHat"""
        
        return flattened_dataset
    

    def save_vtp_cylinder(self) -> None:

        """Creates a .vtp mesh used to generate the slice and saves it."""
        
        R = self._current_slice_offset

        nt = int(2*np.pi*R / self.mesh_element_size)
        nt = max(36, nt); nt = min(int(1e4), nt)   # nt in (36, 1e6) 
        
        nz = int(self.cylinder_height / self.mesh_element_size)   # So to have a square-ish grid
        
        # Calculate H
        H = nz * self.mesh_element_size

        # Create unrotated coordinates
        z_vals = np.linspace(0, H, nz) - H/2
        d_theta = 1e-6      # Small gap around theta=0 for easier surface development
        theta_vals = np.linspace(d_theta, 2*np.pi-d_theta, nt)

        points = [] # Shape (N, 3)
        for z in z_vals:
            for theta in theta_vals:
                points.append([R*np.cos(theta),
                            R*np.sin(theta),
                            z])
        points = np.array(points)

        # Rotate and translate them in place
        rotation_matrix = np.vstack((self.x_axis, self.y_axis, self.z_axis))
        points = points @ rotation_matrix + self.cylinder_center

        # Create .vtp file
        filename = os.path.join(output_path, 
                                self.title,
                                self.vtp_source_name)

        with open(filename, "w") as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
            f.write('  <PolyData>\n')
            total_points = nt * nz
            total_quads = (nz-1)*(nt-1)

            f.write(f'    <Piece NumberOfPoints="{total_points}" NumberOfStrips="0" NumberOfPolys="{total_quads}">\n')

            # Points
            f.write('      <Points>\n')
            f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
            for i in range(points.shape[0]):
                f.write(f'{points[i,0]} {points[i,1]} {points[i,2]} ')
            f.write('\n        </DataArray>\n')
            f.write('      </Points>\n')

            # Quads
            f.write('      <Polys>\n')
            f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">')
            for iz in range(nz-1):
                for it in range(nt-1):
                    p0 = iz*nt + it
                    p1 = iz*nt + it+1
                    p2 = (iz+1)*nt + it+1
                    p3 = (iz+1)*nt + it
                    f.write(f'\n{p0} {p1} {p2} {p3} ')
            f.write('\n        </DataArray>\n')

            f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
            offset = 0
            for _ in range(total_quads):
                offset += 4
                f.write(f'{offset} ')
            f.write('\n        </DataArray>\n')
            f.write('      </Polys>\n')

            f.write('    </Piece>\n')
            f.write('  </PolyData>\n')
            f.write('</VTKFile>\n')
    

    def _update_source(self):

        # Recreate the cylinder vtp mesh with updated radius
        self.save_vtp_cylinder()

        # Update the pipeline
        ReloadFiles(self.vtp_source)  # type: ignore
        

    ### Small deviation from super().create_all_images()
    def create_all_images(self) -> None:
        """
        Creates and saves all images related to the view
        """ 
        super().create_all_images()

        # Delete the .vtp source associated with the view
        os.remove(os.path.join(output_path, 
                               self.title, 
                               self.vtp_source_name))


def Main(Slices_settings: Slices_Settings,
         CaseType_settings: CaseType_Settings,
         caseName: str,
         modelConfig_summary: str) -> None:
    """
    Main entry point for running slicing operations.
    Splits slice offsets into chunks, prepares tasks, and runs them 
    in parallel with timeouts.
    """

    import Utils.domain_setup as domain_setup

    ### Output setup
    global output_path
    output_path = os.path.join(CFDInsight_outputPath, Slices_settings.output_name)
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)


    ### Create source objects (common across views)
    domain, ghost_meshes = domain_setup.domain_setup(CaseType_settings=CaseType_settings,
                                                   Function_settings=Slices_settings)

    
    ### Iterate through all views, generating images from all of them
    for selected_view in Slices_settings.views:

        # Create folders to store results
        for field_properties in Slices_settings.fields:
            os.makedirs(os.path.join(output_path, 
                                     selected_view.title, 
                                     field_properties.title))   

        # Create View objects from settings
        selected_view.build(domain=domain,
                            ghost_meshes=ghost_meshes,
                            caseName=caseName, 
                            modelConfig_summary=modelConfig_summary,
                            fields_properties=Slices_settings.fields,
                            ghost_meshes_properties=Slices_settings.ghost_meshes)

        # Generate all screenshots
        selected_view.create_all_images()

        # Delete object after use
        del selected_view  