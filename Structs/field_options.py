from dataclasses import dataclass, field
from typing import Literal, Optional, List
import numpy as np

""" Generic Field_options class declaration """
@dataclass
class Field_options:
    """
    Defines a field configuration for visualization and analysis.

    Attributes:
        title (str): Descriptive title for the field display (e.g., "U_mean", "Cp").
        expression (Optional[str]): Mathematical expression defining the field
            *(e.g., "U", None, "sqrt(U_X^2 + U_Y^2)")*. 

            *Note:* fields not present in the raw dataset are automatically generated using a `Calculator` filter.
        field_component (Literal["X", "Y", "Z", "Magnitude"]): Specific component
            of the field to visualize.
        source (Literal["POINTS", "CELLS"]): Indicates whether the field is evaluated
            at points or cells.
        representation_type (Literal["Surface", "Surface With Edges", "Feature Edges", "Points", "Points Gaussian"]): 
            Representation type for displaying the field.
        field_min (float): Minimum value for the color scale range.
        field_max (float): Maximum value for the color scale range.
        color_preset (str): Name of the color map preset
            *(e.g., "Cool to Warm", "Jet", "Blue to Red")*.
        num_colors (int): Number of discrete colors used in the color map.
        cbar_tick_stride (int): Interval at which colorbar ticks or labels are placed
            *(e.g., 1 = label every color, 2 = label every couple of colors)*.

    **For Slices:**
    -----
    **background_grid** : *bool* 
        <br>Whether to show background grid in slice views.

    **For Surfaces:**
    -----
    **prepare_for_DeltaSurfaces** : *bool*
        <br>Whether to prepare fields for delta surface comparison.
    
    **delta_field_max** : *float*
        <br>Maximum value for delta field normalization. *(required when``prepare_for_DeltaSurfaces`` is **True**)*.
    """
    
    title: str = None
    expression: Optional[str] = None
    field_component: Literal['X', 'Y', 'Z', 'Magnitude'] = 'Magnitude'
    source: Literal['POINTS', 'CELLS'] = 'POINTS'
    representation_type: Literal['Surface', 
                                 'Surface With Edges', 
                                 'Feature Edges', 
                                 'Points', 
                                 'Points Gaussian'] = 'Surface'
    field_min: float = -1.0
    field_max: float = 1.0
    color_preset: str = 'Jet'
    num_colors: int = 10
    cbar_tick_stride: int = 2

    background_grid: bool = False
    prepare_for_DeltaSurfaces: bool = False
    delta_field_max: Optional[float] = None

    def __post_init__(self):
        """
        Validate field options after initialization.
        
        Performs basic validation to ensure field configuration is consistent.
        Additional validation may be performed by the specific slice/surface options.

        Raises:
            ValueError: If  validation fails
        """

        if self.title == None or self.title == "":
            raise ValueError("The field title must be specified")
    
        if self.expression is not None:
            if self.field_min >= self.field_max:
                raise ValueError("field_min must be less than field_max")
        
        if self.num_colors <= 0:
            raise ValueError("num_colors must be > 0")
        
        if self.cbar_tick_stride <= 0:
            raise ValueError("num_cbar_labels must be > 0")
        
        ### Calculate tick locations
        bounds = np.linspace(self.field_min, self.field_max, 1+self.num_colors)
        self.tick_locs = bounds[np.arange(0, self.num_colors+1, self.cbar_tick_stride)]


        if self.delta_field_max is not None and self.delta_field_max <= 0:
            raise ValueError("delta_field_max must be > 0")
        
        if not self.prepare_for_DeltaSurfaces and self.delta_field_max is not None:
            self.delta_field_max = None
        
        if self.prepare_for_DeltaSurfaces and self.delta_field_max is None:
            raise ValueError("delta_field_max must be specified when prepare_for_DeltaSurfaces is True")
        
        ### Set up computed_by_calculator flag to False
        self.computed_by_calculator = False
        
    def __str__(self) -> str:
        # Start with Field identity (prefer title, fallback to expression)
        display_name = self.title or (self.expression or "<unnamed>")
        string = f"FieldOptions: {display_name}"

        # Add expression if present
        if self.expression is not None:
            string += f"\n\t  Expression: {self.expression}"

        # Add component and data source (POINTS/CELLS)
        string += f"\n\t  Component @ Source: {self.field_component} @ {self.source}"

        # Add representation type
        string += f"\n\t  Representation: {self.representation_type}"

        # Add scalar range
        string += f"\n\t  Range: [{self.field_min:.3g}, {self.field_max:.3g}]"

        # Add color mapping info
        string += f"\n\t  Colormap: {self.color_preset} (N={self.num_colors}, tick_stride={self.cbar_tick_stride})"

        # Add background grid flag
        string += f"\n\t  Background grid: {self.background_grid}"

        # Add DeltaSurfaces preparation flag (and max if enabled)
        string += f"\n\t  DeltaSurfaces prep: {self.prepare_for_DeltaSurfaces}"
        if self.prepare_for_DeltaSurfaces and self.delta_field_max is not None:
            string += f" [delta_field_max: {self.delta_field_max:.3g}]"

        return string
             

""" Special Field_options classes declaration """
@dataclass
class LICField_options:
    """
    Comprehensive configuration for field visualization using Linear Integral Convolution.
    This class presents additional options to "Field_options".

    Attributes:
        title (str): Descriptive title for the field display (e.g., "U_mean", "Cp").
        expression (Optional[str]): Mathematical expression defining the field 
            (e.g., "U", None, "sqrt(U_X^2 + U_Y^2)").
            
            *Note:* fields not present in the raw dataset are automatically generated using a `Calculator` filter.
        field_component (Literal["X", "Y", "Z", "Magnitude"]): Specific component 
            of the field to visualize.
        source (Literal["POINTS", "CELLS"]): Indicates whether the field is evaluated 
            at points or cells.
        representation_type (Literal["Surface", "Surface With Edges", "Feature Edges", "Points", "Points Gaussian"]):
            Representation type for displaying the field.
        field_min (float): Minimum value for the color scale range.
        field_max (float): Maximum value for the color scale range.
        color_preset (str): Name of the color map preset 
            (e.g., "Cool to Warm", "Jet", "Blue to Red").
        num_colors (int): Number of discrete colors used in the color map.
        cbar_tick_stride (int): Interval at which colorbar ticks or labels are placed 
            *(e.g., 1 = label every color, 2 = label every couple of colors)*.
        
    **LIC Attributes**
    ------
    **LIC_input_vectors** : *str*
        Name of the vector field used as the input for the Line Integral Convolution (LIC)
        algorithm. Typically set to a velocity or vorticity vector field.

    **LIC_color_mode** : *Literal['Blend', 'Multiply']*
        Defines how the LIC pattern is combined with the base color map.
        - 'Blend' smoothly mixes the LIC texture with the underlying color.
        - 'Multiply' darkens or lightens the color map based on LIC intensity.

    **LIC_enhance_contrast** : *Literal['Off', 'LIC Only', 'LIC and Color', 'Color Only']* 
        Controls contrast enhancement applied after LIC computation.
        - 'Off' disables any contrast adjustment.
        - 'LIC Only' enhances the convolution texture only.
        - 'LIC and Color' enhances both texture and color.
        - 'Color Only' enhances only the color mapping.

    **LIC_num_steps** : *int*
        Number of integration steps used when computing the LIC pattern.
        Higher values increase smoothness but also computational cost.

    **LIC_step_size** : *float*
        Step length (in dataset units) used for integration along the vector field.
        Smaller values yield finer detail, at the expense of increased computation time.    

    **For Slices:**
    -----
    **background_grid** : *bool* 
        <br>Whether to show background grid in slice views.                   
    """

    title: str = None
    expression: Optional[str] = None
    field_component: Literal['X', 'Y', 'Z', 'Magnitude'] = 'Magnitude'
    source: Literal['POINTS', 'CELLS'] = 'POINTS'
    representation_type: Literal['Surface', 
                                 'Surface With Edges', 
                                 'Feature Edges', 
                                 'Points', 
                                 'Points Gaussian'] = 'Surface'
    field_min: float = -1.0
    field_max: float = 1.0
    color_preset: str = 'Jet'
    num_colors: int = 10
    cbar_tick_stride: int = 2
    
    LIC_input_vectors: str = None 
    LIC_color_mode: Literal['Blend', 'Multiply'] = 'Multiply'
    LIC_enhance_contrast: Literal['Off', 
                                  'LIC Only', 
                                  'LIC and Color',
                                  'Color Only'] = 'LIC Only'
    LIC_num_steps: int = 40
    LIC_step_size: float = 0.5

    background_grid: bool = False

    def __post_init__(self):
        """
        Validate field options after initialization.
        
        Performs basic validation to ensure field configuration is consistent.
        Additional validation may be performed by the specific slice/surface options.

        Raises:
            ValueError: If attributes validation fails
        """

        if self.title == None or self.title == "":
            raise ValueError("The field title must be specified")

        if self.expression is not None:
            if self.field_min >= self.field_max:
                raise ValueError("field_min must be less than field_max")
        
        if self.num_colors <= 0:
            raise ValueError("num_colors must be > 0")
        
        if self.cbar_tick_stride <= 0:
            raise ValueError("num_cbar_labels must be > 0")
        
        ### Calculate tick locations
        bounds = np.linspace(self.field_min, self.field_max, 1+self.num_colors)
        self.tick_locs = bounds[np.arange(0, self.num_colors+1, self.cbar_tick_stride)]
        
        if self.LIC_input_vectors is None:
            raise ValueError("An expression for LIC_input_vectors must be entered")
        if self.LIC_num_steps < 0:
            raise ValueError("num_LIC_steps must be greater than 0")
        if self.LIC_step_size < 0:
            raise ValueError("step_size must be greater than 0")
        
        ### Set up computed_by_calculator flag to False
        self.computed_by_calculator = False
        
    def __str__(self) -> str:
        # Start with Field identity (prefer title, fallback to expression)
        display_name = self.title or (self.expression or "<unnamed>")
        string = f"LICFieldOptions: {display_name}"

        # Add expression if present
        if self.expression is not None:
            string += f"\n\t  Expression: {self.expression}"

        # Add component and data source (POINTS/CELLS)
        string += f"\n\t  Component @ Source: {self.field_component} @ {self.source}"

        # Add representation type
        string += f"\n\t  Representation: {self.representation_type}"

        # Add scalar range
        string += f"\n\t  Range: [{self.field_min:.3g}, {self.field_max:.3g}]"

        # Add color mapping info
        string += f"\n\t  Colormap: {self.color_preset} (N={self.num_colors}, tick_stride={self.cbar_tick_stride})"

        # Add LIC configuration block
        string += f"\n\t  LIC:"
        string += f"\n\t    - Input vectors: {self.LIC_input_vectors}"
        string += f"\n\t    - Blend mode: {self.LIC_color_mode}"
        string += f"\n\t    - Enhance contrast: {self.LIC_enhance_contrast}"
        string += f"\n\t    - Steps / Step size: {self.LIC_num_steps} / {self.LIC_step_size:.3g}"

        # Add background grid flag
        string += f"\n\t  Background grid: {self.background_grid}"

        return string


""" GhostMeshes class declaration """
@dataclass
class GhostMeshes:
    """
    Container for representing mesh geometry used as a "ghost" object in Slices
    and Surfaces visualizations. Useful for providing context to the visualization.

    *Note: the STLs are searched for inside "case_folder_path/constant/trisurface".*

    Attributes:
        filenames (List[str]): 
            List of STL file names to include in slices.
        color (List[float]):
            RGB color triplet in the range [0.0, 1.0].
        opacity (float):
            Opacity factor (0.0 = fully transparent, 1.0 = fully opaque).
    """
    names: List[str]          
    color: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0]) 
    opacity: float = 0.2

    def __str__(self) -> str:
        # Start with container header
        string = "GhostMeshes:"

        # Add number of STL names and preview a few
        n = len(self.names) if self.names is not None else 0
        preview = ", ".join(self.names[:3]) + ("..." if n > 3 else "") if n else "None"
        string += f"\n\t  Mesh files: {n} ({preview})"

        # Add color info (RGB in [0,1])
        if self.color is not None and len(self.color) == 3:
            string += f"\n\t  Color (RGB): [{self.color[0]:.2f}, {self.color[1]:.2f}, {self.color[2]:.2f}]"
        else:
            string += f"\n\t  Color (RGB): <invalid>"

        # Add opacity
        string += f"\n\t  Opacity: {self.opacity:.2f}"

        return string

