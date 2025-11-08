from Utils.shared_imports import *
from Structs.generic_function import Function_Settings
from Structs.generic_view import create_cbar_in_fig
from Functions.Surfaces import Surfaces_Settings

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors

""" Utilities """
def _get_dirs_and_files(path):
    """Returns tuple of (subdirectories, files) using pathlib"""
    p = Path(path)
    dirs = [d.name for d in p.iterdir() if d.is_dir()]
    files = [f.stem for f in p.iterdir() if f.is_file()]
    return dirs, files

def _custom_colormap(colormap_name, num_colors, values_to_ignore):
    """
    Create a custom discrete colormap with `num_colors` and modify it to include transparency for the mid-value.
    """
    cmap = plt.get_cmap(colormap_name)     # Get continuous colormap
    colors = cmap(np.linspace(0, 1, 2*num_colors))   # Extract RGBA colors at discrete locations

    # Declaring indices to which to apply the transparency
    indices_to_remove = np.arange(num_colors-values_to_ignore, 
                        num_colors+values_to_ignore, 
                        1, 
                        dtype=int)

    # Make alpha 0 where the delta approaches 0
    colors[indices_to_remove] = (*background_color, 0.0)

    # Return a new ListedColormap with the modified colors
    return mcolors.ListedColormap(colors), indices_to_remove

def _values_from_greyscale(path_to_image: str, 
                           path_to_geom: str, 
                           minVal: float, 
                           maxVal: float) -> np.ndarray:

    # Try to open the file
    try:
        # rgb to scalar (greyscale) conversion
        img = mpimg.imread(path_to_image)[..., 0] 
        background = mpimg.imread(path_to_geom)[..., 0] 
    except ImportError as e:
        print(f"Error on Import: {e}")
        return None
    
    # If the 2 arrays don't have the same shape, raise a Warning
    if img.shape != background.shape:
        raise Warning(f"The 2 sources have incompatible sizes!")
    
    # Obtain the values from mapping
    a = maxVal - minVal; b = minVal * np.ones(np.shape(img))

    # Do the mapping
    return a * (img - background) + b


""" Settings for Preferences.py file """
@dataclass
class DeltaSurfaces_Settings(Function_Settings):
    """
    Settings for delta surface comparison between different simulation results.
    
    Attributes:
        baseline_path (str):
            Path to the baseline case for comparison. *(has to be absolute!)*
    """

    baseline_path: str

    def __post_init__(self):
        """Initialize with default name for delta surface settings."""
        self.output_name = "DeltaSurfaces"

        # Convert to absolute path
        if self.run:
            print(self.baseline_path)

            # Expand user (~) and trim whitespace, but don't resolve yet
            p_raw = Path(str(self.baseline_path).strip()).expanduser().resolve()

            if not p_raw.is_dir():
                raise IsADirectoryError(f"the baseline path {p_raw} is not valid")

            # Resolve (canonicalize, follow symlinks)
            self.baseline_path = str(p_raw)

            

""" DeltaSurface_View class declaration """
class DeltaSurface_View:
    def __init__(self, 
                 case_name: str, 
                 baseline_name: str, 
                 view_name: str,
                 case_view_path: str, 
                 baseline_view_path: str):
        
        # Get names
        self.case_name = case_name
        self.baseline_name = baseline_name
        self.view_name = view_name

        # Get paths to case and baseline images
        self.case_view_path = case_view_path
        self.baseline_view_path = baseline_view_path

        # Get path to background geometry img
        self.path_to_geomImg = os.path.join(case_view_path, "Geometry.png")
        
        # Get field names
        _, self.fields_list = _get_dirs_and_files(self.case_view_path)
        self.fields_list.remove("Geometry")

        # Create output directory
        self.output_dir = os.path.join(output_path, self.view_name)
        os.makedirs(self.output_dir)

    def create_all_images(self):

        # Cicle through all fields inside "self.fields_list" and create images
        for field_name in self.fields_list:

            # Find original surface field min and max values
            field_settings = next((field for field in surface_fields 
                                   if (field.title == field_name and
                                   field.prepare_for_DeltaSurfaces)), 
                                   None)
            field_min = field_settings.field_min
            field_max = field_settings.field_max
            delta_field_max = field_settings.delta_field_max

            # Map greyscale difference to values
            case_values = _values_from_greyscale(path_to_image=os.path.join(self.case_view_path, f"{field_name}.png"),
                                                 path_to_geom=self.path_to_geomImg,
                                                 minVal=field_min,
                                                 maxVal=field_max)
            
            baseline_values = _values_from_greyscale(path_to_image=os.path.join(self.baseline_view_path, f"{field_name}.png"),
                                                     path_to_geom=os.path.join(self.baseline_view_path, f"Geometry.png"),
                                                     minVal=field_min,
                                                     maxVal=field_max)
            
            difference = case_values - baseline_values

            # Export the image
            self._create_image(field_name=field_name, 
                               values=difference, 
                               field_min=-delta_field_max,
                               field_max=delta_field_max,
                               num_levels=10)

    def _create_image(self,
        field_name: str,
        values: np.ndarray,
        field_min: float,
        field_max: float,
        num_levels: int):

        # Create the colormap
        colormap, indices_to_remove =_custom_colormap(colormap_name="jet",
                                                      num_colors=num_levels,
                                                      values_to_ignore=2)
        
        # Get the complete path to the image
        complete_img_path = os.path.join(output_path, self.view_name, f"Delta_{field_name}")

        # Create plot and load background geometry image
        img = mpimg.imread(self.path_to_geomImg)
        
        # Create figure
        fig = plt.figure(figsize=img_size_in, dpi=img_dpi)  
        ax = fig.add_axes([0, 0, 1, 1])  # fill the figure
        
        # Remove axes
        ax.axis('off')
        
        # Display image
        ax.imshow(img)

        # Overlay the values
        ax.imshow(values, cmap=colormap, alpha=.9, vmin=field_min, vmax=field_max)

        # Add contours
        contour_levels = np.linspace(field_min, field_max, 2*num_levels+1)
        contour_levels = np.delete(contour_levels, indices_to_remove[1:])

        ax.contour(values, levels=contour_levels, 
                   colors='black', linewidths=0.1, alpha=0.5)

        # Create title string
        title = (
            r"$\bf{Case:}$" + f" {self.case_name}$\\bf{{;}}$   "
            + r"$\bf{Baseline:}$" + f" {self.baseline_name}$\\bf{{;}}$\n"
            + r"$\bf{Field:}$" + f" {field_name}$\\bf{{;}}$   "
        )
        
        # Set title
        ax.set_title(title, 
                    linespacing=title_linespacing, 
                    fontsize=title_font_size, 
                    loc='left')

        # Add a colorbar
        create_cbar_in_fig(fig=fig,
                           ax=ax,
                           field_title=f'Delta {field_name}',
                           field_min=field_min,
                           field_max=field_max,
                           cmap='jet',
                           num_colors=10,
                           tick_locs=np.linspace(field_min, 
                                                 field_max, 
                                                 1+num_levels))

    
        ### Save figure and close
        plt.savefig(complete_img_path, 
                    dpi=img_dpi, 
                    bbox_inches="tight",
                    pad_inches=pad_in)
        plt.close()

        
def Main(DeltaSurfaces_settings: DeltaSurfaces_Settings,
         case_Surfaces_settings: Surfaces_Settings,
         case_name: str, 
         baseline_name: str) -> None:
    
    
    # Create output folder and set outputPath as global
    global output_path, surface_fields
    output_path = os.path.join(CFDInsight_outputPath, DeltaSurfaces_settings.output_name, baseline_name)
    surface_fields = case_Surfaces_settings.fields

    # Reset output dir
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    # Get the names of all views
    view_names, _ = _get_dirs_and_files(os.path.join(CFDInsight_outputPath, case_Surfaces_settings.output_name))


    ### Create DeltaSurface_View objects
    views_list: List[DeltaSurface_View] = []

    for view_name in view_names:
        # Create paths to both the current case view and the corresponding baseline view
        case_view_path = os.path.join(CFDInsight_outputPath, "Surfaces", view_name, "for_DeltaSurfaces")
        baseline_view_path = os.path.join(DeltaSurfaces_settings.baseline_path, output_folder_title, 
                                          "Surfaces", view_name, "for_DeltaSurfaces")
        
        # Check if "baseline_view_path" exists, if not raise a warning
        if not os.path.isdir(baseline_view_path):
            raise Warning(f"No \"{view_name}\" view was found in the baseline case!")

        # Create the object
        views_list.append(DeltaSurface_View(case_name=case_name,
                                            baseline_name=baseline_name,
                                            view_name=view_name,
                                            case_view_path=case_view_path,
                                            baseline_view_path=baseline_view_path))


    ### Create images for all views
    for view_obj in views_list:
        view_obj.create_all_images()





