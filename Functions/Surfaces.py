from Utils.shared_imports import *
from Utils.pvserver_utils import *
from Structs.generic_function import Function_Settings
from Structs.generic_view import View
import matplotlib.pyplot as plt
from Structs.field_options import *
from Structs.case_types import CaseType_Settings
from copy import deepcopy


""" Settings for Preferences.py file """
@dataclass
class Surfaces_Settings(Function_Settings):
    """
    Settings for surface-based visualization and analysis.
    
    Attributes:
        run (bool): Flag that determines wether the function has to be run.
        mesh_regions (List[str]): List of mesh region names to include in surfaces.
        ghost_meshes (GhostMeshes): Structure that stores options regarding meshes to display along the result.
        views (List[StandardSurface_View]): List of Surface_View configurations.
        fields (List[Union[Field_options, LICField_options]]): List of field options for visualization.
    """
    mesh_regions: List[str]
    ghost_meshes: GhostMeshes
    views: List["StandardSurface_View"]
    fields: List[Union[Field_options, LICField_options]]

    def __post_init__(self):
        """Initialize with default name for surface settings."""
        self.output_name = "Surfaces" 


""" StandardSurface_View class declaration """
class StandardSurface_View(View):  
    def __init__(self,
                 title: str,
                 camera_position: List[float],
                 focus_position: List[float],
                 viewUp_direction: List[float],
                 interaction_mode: Literal['2D', '3D'] = '3D',
                 parallel_scale: float = 1.0):
        """
        Defines a standard surface view configuration for 3D visualization.
        
        Attributes:
            title (str): Descriptive title for the surface view.
            camera_position ([float, float, float]): Vector defining the camera position.
            focus_position ([float, float, float]): Vector defining the camera focus point.
            viewUp_direction ([float, float, float]): Vector defining the 'up' direction for the camera.
            interaction_mode (Literal['2D', '3D']): String that defines the rendering mode.
            parallel_scale (float): Scaling factor for parallel projection.
                **`Only effective if interaction_mode is set to '2D'`**.
        """

        # Attributes initialization
        super().__init__(title = title,
                         camera_position = camera_position,
                         focus_position = focus_position,
                         viewUp_direction = viewUp_direction,
                         parallel_scale=parallel_scale,
                         interaction_mode=interaction_mode) 


    ### View building
    def build(self, 
              domain,
              ghost_meshes,
              caseName: str, 
              modelConfig_summary: str,
              fields_properties: List[Union[Field_options, LICField_options]],
              ghost_meshes_properties: GhostMeshes):
        

        ### Creating self.delta_field_list
        # fields with prepare_for_DeltaSurfaces flag as True will be cloned into new 
        # fields with 'Grayscale' colormap.
        # The flag will then be set to False in the original field
        fields_properties = deepcopy(fields_properties)     # Deepcopying to avoid interactions between views
        self.delta_field_list: List[Field_options] = []
        for i in range(len(fields_properties)):
            field = fields_properties[i]
            if hasattr(field, "prepare_for_DeltaSurfaces") and field.prepare_for_DeltaSurfaces:
                # Create the clone with 'Grayscale' colormap
                clone_field = deepcopy(field)
                clone_field.color_preset = 'Grayscale'
                clone_field.num_colors = 256

                # Append the cloned field to the list
                self.delta_field_list.append(clone_field)

                # Set the original field flag to False
                field.prepare_for_DeltaSurfaces = False

        # If there were any fields marked with prepare_for_DeltaSurfaces,
        # Append a Geometry screenshot to the list
        if len(self.delta_field_list) > 0:
            self.delta_field_list.append(Field_options(
                                            title="Geometry",
                                            expression=None,
                                            representation_type='Surface',
                                            prepare_for_DeltaSurfaces=True,
                                            delta_field_max=1.0   # Placeholder delta_field_max
                                            )
                                        )

        ### Use parent class build function
        super().build(domain=domain,
                      ghost_meshes=ghost_meshes,
                      caseName=caseName,
                      modelConfig_summary=modelConfig_summary,
                      fields_properties=fields_properties,
                      ghost_meshes_properties=ghost_meshes_properties)
        
    
    ### Source-related
    def _create_source(self, domain):
        return domain
    
        
    ### Image-related    
    def _build_complete_img_path(self, 
                                 output_path: str, 
                                 field_props: Field_options):
        
        # If the self.prepare_for_DeltaSurfaces flag is set to True
        if hasattr(field_props, "prepare_for_DeltaSurfaces") and field_props.prepare_for_DeltaSurfaces:
            img_title = f"{self._current_field.title}.png"
            return os.path.join(output_path, self.title, "for_DeltaSurfaces", img_title)

        # Use different convention otherwise
        img_title = f"{self._screenshot_counter}_{self._current_field.title}.png"
        return os.path.join(output_path, self.title, img_title)


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
        
        # Stop the screenshot refinement if the image is to be used in delta_surfaces
        if hasattr(field_options, "prepare_for_DeltaSurfaces") and field_options.prepare_for_DeltaSurfaces:
            return False

        return True
    

    def create_all_images(self) -> None:
        """
        Creates and saves all images related to the view
        """    
        
        ### Cicle through all fields inside "self.field_list" and create images
        for field_properties in self.field_list:
            # Apply the field to the model inside the RenderView
            self._apply_field(field_properties)

            # Build img output path
            complete_img_path = self._build_complete_img_path(
                                    output_path=output_path,
                                    field_props = field_properties)

            # Create the screenshot
            self._export_image(complete_img_path=complete_img_path,
                               field_options=field_properties)
            
            # Update the counter inside the object
            self._screenshot_counter += 1


        # Hide the ghost meshes for delta screenshots
        Hide(self.ghost_meshes, self.renderView)    # type: ignore

        # Set the background to white
        self.renderView.Background = [1,1,1]

        ### Cicle through all fields inside "self.delta_field_list" and create images
        for field_properties in self.delta_field_list:
            # Apply the field to the model inside the RenderView
            self._apply_field(field_properties)

            # Build img output path
            complete_img_path = self._build_complete_img_path(
                                    output_path=output_path,
                                    field_props = field_properties)

            # Create the screenshot
            self._export_image(complete_img_path=complete_img_path,
                               field_options=field_properties)
            
            # (counter update not needed)


def Main(Surfaces_settings: Surfaces_Settings,
         CaseType_settings: CaseType_Settings,
         caseName: str,
         modelConfig_summary: str) -> None:
    """
    Main entry point for generating surface visualizations.
    Prepares tasks for each view and executes them in parallel with timeouts.
    """

    import Utils.domain_setup as domain_setup
    
    ### Output setup
    global output_path
    output_path = os.path.join(CFDInsight_outputPath, Surfaces_settings.output_name)
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)


    ### Create source objects (common across views)
    domain, ghost_meshes = domain_setup.domain_setup(CaseType_settings=CaseType_settings,
                                                     Function_settings=Surfaces_settings)
    

    ### Iterate through all views, generating images from all of them
    for selected_view in Surfaces_settings.views:
    
        # Create folders to store results
        os.makedirs(os.path.join(output_path, selected_view.title))
        os.makedirs(os.path.join(output_path, selected_view.title, "for_DeltaSurfaces"))

        # Create View objects from settings
        selected_view.build(domain=domain,
                            ghost_meshes=ghost_meshes,
                            caseName=caseName, 
                            modelConfig_summary=modelConfig_summary,
                            fields_properties=Surfaces_settings.fields,
                            ghost_meshes_properties=Surfaces_settings.ghost_meshes)

        # Generate all screenshots
        selected_view.create_all_images()

        # Delete object after use
        del selected_view   