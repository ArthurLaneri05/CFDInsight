from Utils.shared_imports import *
from Structs.field_options import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcmp
import time
from PIL import Image, ImageFile


class View(ABC):
    def __init__(self,
                 title: str,
                 camera_position: List[float],
                 focus_position: List[float],
                 viewUp_direction: List[float],
                 interaction_mode: Literal['2D', '3D'],
                 parallel_scale: float):
        """
        Initialize a view object with camera, interaction, and display parameters.

        Args:
            title (str): Title of the view.
            camera_position (List[float]): Camera position coordinates.
            focus_position (List[float]): Focus (target) position coordinates.
            viewUp_direction (List[float]): Orientation of the camera's "up" direction.
            interaction_mode (Literal['2D','3D']): Rendering interaction mode.
            parallel_scale (float): Parallel projection scaling factor.
        """
        # Attributes initialization
        self.title = title
        self.camera_position = camera_position
        self.focus_position = focus_position
        self.viewUp_direction = viewUp_direction  
        self.interaction_mode = interaction_mode
        self.parallel_scale = parallel_scale

        # Source-related
        self.source = None
        self.ghost_meshes = None

        # Field-related
        self.field_list: List[Field_options] = []
        self._current_field: Field_options = None
        self._current_colormap: mcolors.ListedColormap = None

        # Display-related
        self.caseName: str = None
        self.configSummary: str = None

        # Utils-related
        self._screenshot_counter = 0

    
    ### View building
    def build(self, 
              domain,
              ghost_meshes,
              caseName: str, 
              modelConfig_summary: str,
              fields_properties: List[Union[Field_options, LICField_options]],
              ghost_meshes_properties: GhostMeshes
              ) -> None:
        """
        Build the visualization view by setting up fields, source, and render view.

        Args:
            domain: Geometric domain or mesh object.
            ghost_meshes: Stacked meshes of ghost stls.
            caseName (str): Name of the case to display.
            modelConfig_summary (str): Text summary of the model configuration.
            fields_properties (List[Field_options]): List of field visualization options.
            ghost_meshes_properties (GhostMeshes): auxiliary meshes visualization options.
        """
        # Field-related
        self.field_list = fields_properties
        self.ghost_meshes_props = ghost_meshes_properties
        
        # Display-related
        self.caseName = caseName
        self.configSummary = modelConfig_summary
        
        # Sources creation
        self.source = self._create_source(domain)
        self.ghost_meshes = ghost_meshes

        # View creation
        self.renderView = None
        self._create_RenderView_and_Display()


    ### Source-related
    @abstractmethod
    def _create_source(self, domain) -> None:
        """
        Create the data source object for the view.

        Args:
            domain: Geometric domain or mesh object.

        Returns:
            None
        """
        pass


    ### Display-related
    def _create_RenderView_and_Display(self) -> None:
        """
        Create the RenderView and configure its display settings.
        
        Returns:
            None
        """
        # Create renderView
        self.renderView = CreateView('RenderView')    # type: ignore

        # Create display
        self.display = GetRepresentation(self.source,    # type: ignore 
                                         view=self.renderView)

        # Show ghost meshes too, no need to save the display object
        if self.ghost_meshes:
            ghost_meshes_display = GetRepresentation(self.ghost_meshes,    # type: ignore 
                                                     view=self.renderView)
            
            # Edit color and opacity of the meshes
            ghost_meshes_display.Set(
                Opacity = self.ghost_meshes_props.opacity,
                AmbientColor=self.ghost_meshes_props.color,
                DiffuseColor=self.ghost_meshes_props.color,
            )

        # Specify renderView settings
        self.renderView.InteractionMode = self.interaction_mode
        if self.interaction_mode == '2D':
            self.renderView.CameraParallelProjection = 1
            self.renderView.CameraParallelScale = self.parallel_scale

        self.renderView.CameraPosition = self.camera_position
        self.renderView.CameraFocalPoint = self.focus_position
        self.renderView.CameraViewUp = self.viewUp_direction
        self.renderView.UseColorPaletteForBackground = 0
        self.renderView.Background = background_color
        self.renderView.OrientationAxesVisibility = 0   # Hide the axes
        
        # For faster rendering
        self.renderView.EnableRenderOnInteraction = 0  # Disable re-rendering

        # Fix camera in place
        # self.renderView.ResetCameraOnVisibilityChange = 0

        # Disable raytracing
        self.renderView.EnableRayTracing = 0


    def _apply_field(self, field_properties: Union[LICField_options, Field_options]) -> None:
        """
        Apply a given field to the render view and configure its representation.

        Args:
            field_properties (LICField_options, Field_options): Field configuration options.

        Returns:
            None
        """
        # Set "self._current_field"
        self._current_field = field_properties

        # Display the model according to field_properties
        component = field_properties.field_component
        expression = (field_properties.expression if not field_properties.computed_by_calculator 
                      else field_properties.title)
        eval_source = field_properties.source
        repr_type = field_properties.representation_type

        self.display.SetRepresentationType(repr_type)

        if expression is None:
            ColorBy(self.display, None)    # type: ignore
        else:
            ColorBy(self.display, (eval_source, expression, component))    # type: ignore
        

        # Special treatment for LICField_options
        if isinstance(field_properties, LICField_options):
            self.display.SetRepresentationType('Surface LIC')
            self.display.SelectInputVectors = [eval_source, field_properties.LIC_input_vectors]
            self.display.ColorMode = field_properties.LIC_color_mode
            self.display.EnhanceContrast = field_properties.LIC_enhance_contrast
            self.display.NumberOfSteps = field_properties.LIC_num_steps
            self.display.StepSize = field_properties.LIC_step_size
  
        # Set up Colormap for fields with an expression (all except Geometry/Mesh) 
        self._current_colormap = None
        if expression is not None: 
            # Rescale transfer functions and apply preset and discretize colors
            lut = GetColorTransferFunction(expression)    # type: ignore
            lut.RescaleTransferFunction(field_properties.field_min, field_properties.field_max)
            lut.ApplyPreset(field_properties.color_preset, True)
            lut.NumberOfTableValues = field_properties.num_colors

            # Translate the ParaView colormap into a ListedColormap
            self._current_colormap = extract_discrete_cmap_from_lut(lut, 
                                                                    field_properties.num_colors)


    ### Image Creation-related
    def _export_image(self,
                  complete_img_path: str,
                  field_options: Union[LICField_options, Field_options]
                  ) -> None:
        """
        Export the current view as an image, optionally refining it with
        titles, labels, and function-specific adjustments.

        Args:
            complete_img_path (str): Directory where the image will be saved.
            field_options (LICField_options, Field_options): Field options.

        Returns:
            None
        """

        # Compute target resolution
        w_px = int(img_size_in[0]*img_dpi)
        h_px = int(img_size_in[1]*img_dpi)
        
        # Write ParaView shot to a temporary raw file to avoid read-while-write hazards
        raw_path = complete_img_path.replace(".png", "_raw.png")

        # Ensure the view is up-to-date before saving (important for screen-space effects)
        self.renderView.StillRender()   

        ### Save raw screenshot (no legends)
        SaveScreenshot(    # type: ignore
            raw_path,
            view=self.renderView,
            ImageResolution=[w_px, h_px],
            CompressionLevel=raw_img_compression_level
        )


        ### Helper function
        def _read_png(path: str, sleep_s: float):
            ImageFile.LOAD_TRUNCATED_IMAGES = False

            while True:
                try:
                    with Image.open(path) as im:
                        im.load()  # force decoding
                        return np.array(im)
                    
                except:
                    time.sleep(sleep_s)

        
        ### Refine screenshot
        # Create figure
        fig, ax = plt.subplots(figsize=img_size_in, dpi=img_dpi)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Remove axes
        ax.axis('off')
        
        # Display image
        img = _read_png(path=raw_path, sleep_s=1)
        ax.imshow(img)


        ### Introduce function-specific settings
        proceed_flag = self._additional_img_ref(field_options=field_options,
                                                fig=fig,
                                                ax=ax,
                                                img=img)
        

        ### Interrupt the screenshot refinement process if needed
        if not proceed_flag:
            plt.close()
            # Rename the raw file
            os.rename(src=raw_path, dst=complete_img_path)
            return


        ### Create title
        title = (
            r"$\bf{Case Name:}$" + f" {self.caseName}$\\bf{{;}}$   "
            + r"$\bf{Field:}$" + f" {self._current_field.title}$\\bf{{;}}$\n"
            + self.configSummary
        )

        ax.set_title(title, 
                    linespacing=title_linespacing, 
                    fontsize=title_font_size, 
                    loc='left')
                

        ### Add a colorbar (if self._current_colormap is not None)
        if self._current_colormap is not None:
            create_cbar_in_fig(fig=fig,
                               ax=ax,
                               field_title=self._current_field.title,
                               field_min=self._current_field.field_min,
                               field_max=self._current_field.field_max,
                               cmap=self._current_colormap,
                               num_colors=self._current_field.num_colors,
                               tick_locs=self._current_field.tick_locs
                               )
        

        ### Save figure and close
        plt.savefig(complete_img_path, 
                    dpi=img_dpi, 
                    bbox_inches="tight",
                    pad_inches=pad_in)
        plt.close()

        # Clean up the raw file if present
        os.remove(raw_path)


    @abstractmethod
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
        pass 


    @abstractmethod
    def create_all_images(self, output_path: str) -> None:
        """
        Creates and saves all images related to the view
        Args:
            output_path (str): Base directory where the images should be saved.
        
        Returns:
            None
        """  
        pass
                

""" Utility that creates a colorbar in a figure """
def create_cbar_in_fig(fig: plt.Figure,
                       ax: plt.Axes,
                       field_title: str,
                       field_min: float,
                       field_max: float,
                       cmap: Union[str, mcolors.Colormap],
                       num_colors: int,
                       tick_locs: np.ndarray) -> None:
    """Create a horizontal colorbar under a given axes.

    Builds a horizontal colorbar aligned to the width of `ax`, placed just
    below it inside the same figure. The colormap can be provided either as a
    Matplotlib colormap **name** (``str``) or as a ``Colormap`` object. When a
    name is given, a discrete colormap with ``num_colors`` bins is created so
    that ticks and color steps align with a ``BoundaryNorm``.

    Args:
        fig (matplotlib.figure.Figure): Target figure.
        ax (matplotlib.axes.Axes): Axes the colorbar should align under.
        field_title (str): Label shown under the colorbar.
        field_min (float): Minimum data value mapped to the colormap.
        field_max (float): Maximum data value mapped to the colormap.
        cmap (str | matplotlib.colors.Colormap): Colormap name (e.g., ``"viridis"``)
            or an existing ``Colormap``/``ListedColormap`` instance. If a name is
            provided, it is discretized to ``num_colors`` bins.
        num_colors (int): Number of discrete color bins used for both the colormap
            and the ``BoundaryNorm``.
        tick_locs (numpy.ndarray): Tick locations in data units along the colorbar.

    Returns:
        None: This function adds artists to the figure and has no return value.

    Raises:
        ValueError: If ``field_min >= field_max`` or ``num_colors < 1``.
        TypeError: If ``cmap`` is neither a ``str`` nor a ``Colormap`` instance.

    Notes:
        - The function creates an auxiliary axes via ``fig.add_axes`` positioned
          based on ``ax.get_position()``; it does not use ``fig.colorbar`` layout
          management.
        - This function expects the global variables ``tick_font_size`` and
          ``label_font_size`` to be defined in the calling scope.

    Example:
        >>> fig, ax = plt.subplots()
        >>> _ = ax.imshow(np.random.rand(10, 10), vmin=-1, vmax=1)
        >>> create_cbar_in_fig(fig, ax, "Cp", -1.0, 1.0, cmap="viridis",
        ...                    num_colors=11, tick_locs=np.linspace(-1, 1, 5))
        >>> plt.show()
    """
    ### basic validation
    if field_min >= field_max:
        raise ValueError("field_min must be < field_max.")
    if num_colors < 1:
        raise ValueError("num_colors must be >= 1.")

    ### auxiliary axes placement
    ax_pos = ax.get_position()  ### bounding box of main plot in figure coords
    cbar_padding = 0.02         ### fraction of ax width used as left/right padding
    cbar_thickness = 0.02       ### fraction of ax height used as colorbar thickness

    cax = fig.add_axes([
        ax_pos.x0 + cbar_padding * ax_pos.width,                 ### left
        ax_pos.y0 - 0.01 - cbar_thickness,                       ### bottom
        (1 - 2 * cbar_padding) * ax_pos.width,                   ### width
        cbar_thickness                                           ### height
    ])

    ### colormap & normalization
    bounds = np.linspace(field_min, field_max, 1 + num_colors)

    if isinstance(cmap, str):
        cmap_obj = mcmp.get_cmap(cmap, num_colors)  ### discretize into N bins
    elif isinstance(cmap, mcolors.Colormap):
        cmap_obj = cmap
    else:
        raise TypeError("`cmap` must be a str or a matplotlib.colors.Colormap.")

    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=num_colors)
    sm = mcmp.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])  ### silence warnings in some Matplotlib versions

    ### colorbar creation
    cbar = plt.colorbar(cax=cax, mappable=sm, orientation='horizontal')

    ### ticks & style
    cbar.set_ticks(tick_locs)
    cbar.ax.tick_params(size=0)
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(labelsize=tick_font_size, pad=1)
    cbar.set_label(field_title, fontsize=label_font_size, labelpad=1)

    ### remove outline for a cleaner look
    cbar.outline.set_visible(False)


""" Utility to extract normalized colormap from a ParaView LUT """
def extract_discrete_cmap_from_lut(lut, N: int) -> mcolors.ListedColormap:
    """
    Utility to extract a discretized ListedColormap from a look-up table.

    Parameters
    ----------
    lut : object
        Must have attribute RGBPoints = [x0, r0, g0, b0, x1, r1, g1, b1, ...]
    N : int
        Number of discrete colors.

    Returns
    -------
    mcolors.ListedColormap
    """

    # Extract and reshape
    rgb_points = lut.RGBPoints
    pts = [(rgb_points[i], rgb_points[i+1], rgb_points[i+2], rgb_points[i+3]) 
           for i in range(0, len(rgb_points), 4)]

    xmin, xmax = pts[0][0], pts[-1][0]
    cdict = {"red": [], "green": [], "blue": []}

    # Build dictionary for continuous colormap
    for x, r, g, b in pts:
        t = (x - xmin) / (xmax - xmin)
        cdict["red"].append((t, r, r))
        cdict["green"].append((t, g, g))
        cdict["blue"].append((t, b, b))

    # Create continuous version first
    continuous_cmap = mcolors.LinearSegmentedColormap("", cdict)

    # Sample it at N evenly spaced points to get discrete colors
    sampled_colors = continuous_cmap(np.linspace(0, 1, N))

    # Return discrete cmap
    return mcolors.ListedColormap(sampled_colors)