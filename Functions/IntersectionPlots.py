from Utils.shared_imports import *
from Structs.generic_function import Function_Settings
from Structs.field_options import *
from Structs.case_types import CaseType_Settings
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from Utils.vector_manipulation_utils import gram_schmidt_ortogonalization

""" Settings for Preferences.py file """
@dataclass
class IntersectionPlots_Settings(Function_Settings):
    """
    A visualization configuration for intersection plots.

    Parameters
    ----------
    run: bool
        Flag that determines whether the function should be executed.
        If set to True, the corresponding function or pipeline will be run;
        otherwise, it will be skipped.
    mesh_regions : List[str]
        Names of mesh regions (e.g. 'wing_surface', 'fuselage') to extract data from.
    plots : List[Union["IntersectionPlane_Plot", "IntersectionCylinder_Plot"]]
        List of plot configurations, one for each geometric intersection.
    field_tuples : List[Tuple[Field_options, ...]]
        Each tuple groups multiple field options that will be displayed
        together in the same plot. Different tuples correspond to separate plots.
    """
   
    mesh_regions: List[str]
    plots: List[Union["IntersectionPlane_Plot",
                      "IntersectionCylinder_Plot"]]
    field_tuples: Iterable[Union[Field_options, 
                                 Tuple[Field_options, ...]]]

    def __post_init__(self):
        """Initialize with default name for intersection plot settings."""
        self.output_name = "IntersectionPlots" 

        # Transforms elements such as (field) into (field,) in self.field_tuples
        self.field_tuples = _fix_tuples(self.field_tuples)

        # Create standard self.fields (essential for domain_setup()!)
        self.fields = _unpack_field_options(self.field_tuples)

        
""" Utils to manage field_tuples """
def _fix_tuples(field_tuples: Iterable[Union[Field_options, Tuple[Field_options, ...]]]
                ) -> List[Tuple[Field_options,...]]:
    
    output: List[Tuple[Field_options,...]] = []
    for g in field_tuples:
        if isinstance(g, Field_options):
            output.append((g,))        # single item becomes tuple
        else:
            output.append(g)        # tuple remains unchanged

    return output


def _unpack_field_options(
        field_tuples: List[Tuple[Field_options,...]]
    ) -> List[Field_options]:

    output: List[Field_options] = []
    for g in field_tuples:
            output.extend(g)        # tuple

    return output


""" Generic _Intersection_Plot class declaration """
class _Intersection_Plot(ABC):
    def __init__(self,
                 title: str,
                 offsets_array: np.ndarray):
        
        # Attributes initialization
        self.title = title
        self.offsets_array = offsets_array
        self.current_plot_offset = 0.0     # Placeholder

        # Data-related
        self._refined_data: Dict[str, np.ndarray] = {}
        self._geom_graph_x_array: np.ndarray = None 
        self._geom_graph_y_array: np.ndarray = None

        # Source-related
        self.source = None

        # Field-related
        self.field_tuple_list: List[Field_options] = []

        # Display-related
        self.caseName: str = None
        self.configSummary: str = None

        # Utils-related
        self._screenshot_counter = 0


    ### Plot building
    def build(self,
              domain,
              caseName: str,
              modelConfig_summary: str,
              plot_titles: List[str],
              field_props_tuples: List[Tuple[Field_options,...]]
              ) -> None:
        
        # Field-related
        self.field_tuple_list = field_props_tuples

        # Unpack all fields names
        self._all_fields = _unpack_field_options(field_props_tuples)
        
        # Display-related
        self._plot_titles = plot_titles
        self.caseName = caseName
        self.configSummary = modelConfig_summary
        
        # Source creation
        self.source = self._create_source(domain)


    ### Source-related
    @abstractmethod
    def _create_source(self, domain) -> None:
        """
        Create the data source object for the plot.

        Args:
            domain: Geometric domain or mesh object.

        Returns:
            None
        """
        pass


    @abstractmethod
    def _update_source(self) -> None:
        pass
    

    ### Data-related
    def _get_data_at_intersection(self, csv_file_path:str) -> int:

        """ Extracts data @ intersection. 
            Returns True if an intersection was found, False if not"""

        ### Save raw data in csv file
        # self.source.UpdatePipeline()
        SaveData(csv_file_path,     # type: ignore
                 proxy=self.source,
                 Precision=15)  
        

        ### Reset self._refined_data
        self._refined_data: Dict[str, np.ndarray] = {}


        ### Read csv file and refine data
        if not os.path.isfile(csv_file_path):
            raise FileNotFoundError(f"CSV not found: {csv_file_path}")

        try:
            # Fill the list of each key using the data from the dat file (if possible)
            with open(csv_file_path, 'r') as f:
                
                ### Extract headers
                header_line = next(f)
                headers = header_line.strip().replace('"',"").split(",") 

                ### Check that there is data at the intersection
                if len(headers) < 3:
                    return False
                
                ### Refine headers (first line of file)
                for i in range(len(headers)):
                    # Replace :0, :1, :2 with _X, _Y, _Z
                    headers[i] = headers[i].replace(':0', '_X').replace(':1', '_Y').replace(':2', '_Z')


                ### Create and fill the raw_data dictionary
                raw_data = {k: [] for k in headers}
                
                for line in f:
                    values = line.strip().split(",")
                    for i, k in enumerate(raw_data.keys()):
                        raw_data[k].append(float(values[i]))

                # Translate the lists into np arrays
                for k in raw_data.keys():
                    raw_data[k] = np.array(raw_data[k])

                
                ### Generate refined dictionary where magnitudes of fields 
                #   that have field_component = 'Magnitude' are automatically 
                #   calculated, and unnecessary fields are discarded
                self._refined_data.update({'Points' : 
                                           np.column_stack((raw_data['Points_X'],
                                                            raw_data['Points_Y'],
                                                            raw_data['Points_Z']))
                                           })
            

                for field in self._all_fields:

                    # Add the field to _refined_data using its title as key
                    key = field.title

                    # Select the values
                    values: np.ndarray = None
                    if field.field_component == 'Magnitude':
                        if key in raw_data.keys():
                            values = raw_data[key]
                        
                        else:
                            try:
                                # Auto-compute the magnitude
                                values = np.sqrt(raw_data[f"{key}_X"]**2 + 
                                                 raw_data[f"{key}_Y"]**2 + 
                                                 raw_data[f"{key}_Z"]**2)
                            except KeyError:
                                raise Warning("Error in processing 'Cp'")
                            
                    
                    # Select the component otherwise
                    else:
                        values = raw_data[f"{key}_{field.field_component}"]

                    # Add the item to the dict
                    self._refined_data.update({key : values})

                
                # ### Sort data
                # order = cluster_points(self._refined_data['Points'])
                # for k in self._refined_data.keys():
                #     self._refined_data[k] = self._refined_data[k][order]
        except Exception as e:
            raise RuntimeError(f"Error processing {csv_file_path}: {str(e)}")
        
        ### Return True as it all went well
        return True
    
    def _export_refined_data(self, csv_file_path:str) -> None:

        """ Exports the refined data into a csv file inside /refined_csv_data"""

        # Build utility dict
        all_data = {'GraphX_m': self._geom_graph_x_array,
                    'GraphY_m': self._geom_graph_y_array,
                    **self._refined_data}    
        all_data.pop('Points')
 
        N_rows = self._geom_graph_x_array.shape[0]

        with open(csv_file_path, "x", encoding="utf-8") as f:
            # write header
            f.write(",".join(all_data.keys()) + "\n")

            # write rows
            for i in range(N_rows):
                row_vals: List[float] = []
                for key in all_data.keys():
                    row_vals.append(str(all_data[key][i]))

                f.write(",".join(row_vals) + "\n")

    @abstractmethod
    def _get_graph_axes_arrays(self) -> Tuple[List[np.ndarray],
                                              np.ndarray]:
        """
        Manipulate the points inside self._refined_data 
        to create a list of x axis arrays and a single y axis array.

        Returns:
            geom_graph_x_axes (List[np.ndarray]): List containing the x axis 
                values and optionally values for an auxiliary x axis for the geometry plot.
            geom_graph_y_axis (np.ndarray): Numpy array containing the y axis 
                values for the geometry plot.
        """
        pass
            

    ### Image-related
    def _export_plot(self,
                     complete_plot_path: str,
                     plot_fields: Tuple[Field_options]
                     ) -> None:
        """
        Export the data at the intersection, both via a csv file and a plot image.

        Args:
            complete_plot_path (str): Complete directory where the image will be saved.
            plot_fields (Tuple[Field_options]): Tuple of all the fields to be displayed in the plot

        Returns:
            None
        """

        # Calculate number of subplots and their size (enforce 2:1 AR)
        subplot_size_in = (img_size_in[0], img_size_in[0] * 0.4)
        N_field_subplots = len(plot_fields)


        ### Geometry
        geom_x = self._geom_graph_x_array,
        geom_y = self._geom_graph_y_array
        

        ### Create figure and subplots
        fig_height_in = (N_field_subplots + 1) * subplot_size_in[1]
        fig = plt.figure(dpi=img_dpi,
                         figsize=(subplot_size_in[0], 
                                  fig_height_in))

        axs: List[plt.Axes] = fig.subplots(nrows=N_field_subplots+1, sharex=True)


        ### Enforce subplots spacing and dimensions
        for ax in axs:
            ax.set_box_aspect(subplot_size_in[1]/subplot_size_in[0])

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, 
                            hspace=0.05)


        ### Set fontsizes etc.
        for ax in axs:
            ax.tick_params(axis='both',
                           labelsize=tick_font_size, 
                           pad=3,
                           size=0)
         
            ax.grid(True, alpha=0.2)    # Set grid


        ### Display title
        fields_str = ", ".join(f"{field.title}" for field in plot_fields)
        title = (
            rf"$\bf{{Case\ Name:}}$ {self.caseName} $\bf{{;}}$   "
            rf"$\bf{{Fields:}}$ " + fields_str + r"$\bf{{;}}$" + "\n"
            + self.configSummary)

        axs[0].set_title(title, 
                         linespacing=title_linespacing, 
                         fontsize=title_font_size, 
                         loc='left')

        
        ### Create bottom "geometry" subplot
        ax_geom: plt.Axes = axs[-1]
        ax_geom.scatter(geom_x, geom_y, 
                        s=2,
                        color="black")
        ax_geom.set_aspect('equal', adjustable='datalim')
        
        # Add margins 
        ax_geom.margins(x=0.05, y=0.15)

        # Add 2nd xaxis (if needed)
        if isinstance(self, IntersectionCylinder_Plot):
            R = self.current_plot_offset
            arclength2deg = lambda l: l/R *180/np.pi
            deg2arclength = lambda d: d * np.pi/180 * R

            secax = ax.secondary_xaxis('bottom', functions=(arclength2deg, deg2arclength))
            secax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}°"))

            secax.tick_params(axis='both',
                              direction='in',
                              labelsize=tick_font_size, 
                              pad=6 + tick_font_size,
                              size=4)

        # Add offset label
        ax.annotate(xy=(0,1),
                    xycoords='axes fraction',
                    xytext=(offset_label_spacing_in*img_dpi, 
                            -offset_label_spacing_in*img_dpi),
                    textcoords='offset pixels',
                    text=r"$\bf{Geometry}:$" + f" Offset={self.current_plot_offset:.3f}m", 
                    fontsize=title_font_size,
                    ha="left", va="top")
        

        ### Create stacked field plots
        for i in range(N_field_subplots):
            ax = axs[i]

            # Field to be plotted
            plot_field = plot_fields[i]
            
            # Set up tick locations
            ax.yaxis.set_ticks(plot_field.tick_locs)

            # Set up y axis limits
            field_delta = plot_field.field_max-plot_field.field_min
            ax.set_ylim(plot_field.field_min - 0.15*field_delta,
                        plot_field.field_max + 0.15*field_delta)

            # Plot the fields
            ax.scatter(geom_x, self._refined_data[plot_field.title],
                       s=2,
                       color=f"C{i}")

            # Set title
            title_corrected = plot_field.title.replace('_', r'\_')

            ax.annotate(xy=(0,1),
                        xycoords='axes fraction',
                        xytext=(offset_label_spacing_in*img_dpi, 
                                -offset_label_spacing_in*img_dpi),
                        textcoords='offset pixels',
                        text=rf"$\bf{{{title_corrected}}}$", 
                        fontsize=title_font_size,
                        ha="left", va="top")

            
        ### Save figure and close
        plt.savefig(complete_plot_path, 
                    dpi=img_dpi, 
                    bbox_inches="tight",
                    pad_inches=pad_in)
        plt.close()


    def create_all_plots(self, output_path: str) -> None:
        """
        Creates and saves all plots related to the object
        Args:
            output_path (str): Base directory where the images should be saved.
        
        Returns:
            None
        """  

        # Init void intersections counter
        N_void_intersections = 0

        # Cicle through all offset_values
        for offset in self.offsets_array:
            
            ### Update "self.current_slice_offset" and update object
            self.current_plot_offset = offset
            self._update_source()
            
            
            ### Get the path in which the csv file has to be saved
            csv_title = f"{self._screenshot_counter}_{self.current_plot_offset:.3f}m.csv"

            # Build csv outputs paths
            raw_csv_path = os.path.join(output_path, self.title, "raw_csv_data", csv_title)
            refined_csv_path = os.path.join(output_path, self.title, "refined_csv_data", csv_title)

            ### Create the raw csv file and write its refined 
            #   content inside self._refined_data
            flag = self._get_data_at_intersection(csv_file_path=raw_csv_path)

            # Update counter if the flag is False
            if not flag:
                N_void_intersections +=1


            ### Proceed only if data at the intersection is present
            if self._refined_data:

                ### Get axes arrays
                (self._geom_graph_x_array, 
                self._geom_graph_y_array) = self._get_graph_axes_arrays()

                ### Save results in csv file
                self._export_refined_data(refined_csv_path)

                ### Create a plot for each plot_title and field_tuple
                for plot_title, field_tuple in zip(self._plot_titles, self.field_tuple_list):

                    # Build plot output path
                    img_title = f"{self._screenshot_counter}_{self.current_plot_offset:.3f}m.png"
                    complete_plot_path = os.path.join(output_path, self.title, "plots", plot_title, img_title)

                    # Export plot
                    self._export_plot(complete_plot_path=complete_plot_path,
                                    plot_fields=field_tuple)
                
            ### Update counter
            self._screenshot_counter += 1


        ### Report any void intersections
        if N_void_intersections > 0:
            print(f"\tWarning on {self.title}: {N_void_intersections} intersections were empty")

            
""" IntersectionPlane_Plot class declaration """
class IntersectionPlane_Plot(_Intersection_Plot):  
    def __init__(self,
                 title: str,
                 plane_normal: List[float],
                 graph_x_axis: List[float],
                 offsets_array: np.ndarray):
        
        """
        Defines a plot along the intersection between the mesh and a plane.

        Attributes:
            title (str): Descriptive title for the plot.
            plane_normal ([float, float, float]): 3D vector defining the normal direction of the intersection plane.
            graph_x_axis ([float, float, float]): Reference direction **in the plane** used as the plot abscissa.
                + *The x_axis vector is automatically orthogonalized with z_axis using 'Gram-Schmidt' orthogonalization.*
            offsets_array (np.ndarray): Array of plane offset values in **meters**.
        """

        # Attributes initialization
        (self.x_axis, 
         self.y_axis, 
         self.normal_vector) = gram_schmidt_ortogonalization(graph_x_axis, plane_normal)

        # Parent class initialization
        super().__init__(title=title,
                         offsets_array=offsets_array)


    ### Source-related
    def _create_source(self, domain):
        """
        Intersect the input returning the PlotOnIntersectionCurves object.
        """

        plot_on_int = PlotOnIntersectionCurves(registrationName='source',  # type: ignore
                                               Input=domain)
        plot_on_int.SliceType = 'Plane'

        # Initialize the plane object
        plot_on_int.SliceType.Set(
            Origin=[0,0,0],
            Normal=self.normal_vector,
            Offset=0,
        )
        
        return plot_on_int


    def _update_source(self):
        # Can't use offset because the program has a bug
        self.source.SliceType.Set(
            Origin=[comp*self.current_plot_offset 
                    for comp in self.normal_vector]
        )

        # Update the pipeline
        # UpdatePipeline(proxy=self.source)   # type: ignore
        
    
    ### Data-related
    def _get_graph_axes_arrays(self) -> Tuple[List[np.ndarray],
                                              np.ndarray]:
        
        graph_x = self._refined_data['Points'] @ self.x_axis
        graph_y = self._refined_data['Points'] @ self.y_axis
        
        return graph_x, graph_y
        

""" IntersectionCylinder_Plot class declaration """
class IntersectionCylinder_Plot(_Intersection_Plot):  
    def __init__(self,
                 title: str,
                 center: List[float],
                 x_axis: List[float],
                 z_axis: List[float],
                 offsets_array: np.ndarray):
        
        """
        Defines a plot along the intersection between the mesh and a cylinder.

        Attributes:
            title (str): Descriptive title for the plot
            center ([float, float, float]): Vector defining the cylinder's center position.
            x_axis (list[float, float, float]): Direction vector of the cylinder’s local x-axis. 
                + *The surface region aligned with +x_axis is mapped to the centerline of the unwrapped (developed) plot.*
                + *The x_axis vector is automatically orthogonalized with z_axis using 'Gram-Schmidt' orthogonalization.*
            z_axis ([float, float, float]): Vector defining the cylinder's axis.
            offsets_array (np.ndarray): Array of cylinder radii values in **meters** *`(negative values NOT allowed!)`*.
        """

        # Check that 0 is not present in offsets_array
        if 0.0 in offsets_array:
            raise ValueError("0.0 must not be present in offsets_array")

        # Attributes initialization
        self.center = center

        # Create ortonormal reference frame
        (self.x_axis, 
         self.y_axis, 
         self.z_axis) = gram_schmidt_ortogonalization(x_axis, z_axis)

        # Parent class initialization
        super().__init__(title=title,
                         offsets_array=offsets_array)


    ### Source-related
    def _create_source(self, domain):
        """
        Intersect the input returning the PlotOnIntersectionCurves object.
        """

        plot_on_int = PlotOnIntersectionCurves(registrationName='source',  # type: ignore
                                               Input=domain)
        plot_on_int.SliceType = 'Cylinder'

        # Initialize the plane object
        plot_on_int.SliceType.Set(
            Center=self.center,
            Axis=self.z_axis,
            Radius=1.0,     # Placeholder value
        )

        return plot_on_int


    def _update_source(self):
        self.source.SliceType.Set(
            Radius=self.current_plot_offset,
        )

        # Update the pipeline
        UpdatePipeline(proxy=self.source)   # type: ignore
    

    ### Data-related
    def _get_graph_axes_arrays(self) -> Tuple[List[np.ndarray],
                                              np.ndarray]:
        
        # Define distances from center
        dists_from_center = (self._refined_data['Points'] - 
                             np.array(self.center))
        
        # Define graph_y
        graph_y = dists_from_center @ self.z_axis

        # Define radial vectors
        radial_vectors = dists_from_center - graph_y[:, None] * self.z_axis

        # Define graph_theta (in radians)
        graph_theta = np.arctan2(radial_vectors @ self.y_axis,
                                 radial_vectors @ self.x_axis)

        # Create graph_x
        graph_x = graph_theta * self.current_plot_offset
        
        return graph_x, graph_y


def Main(IntersectionPlots_settings: IntersectionPlots_Settings,
         CaseType_settings: CaseType_Settings,
         caseName: str,
         modelConfig_summary: str) -> None:
    """
    Main entry point for running intersection operations.
    Splits intersection offsets into chunks, prepares tasks, and runs them 
    in parallel with timeouts.
    """

    import Utils.domain_setup as domain_setup

    ### Output setup
    output_path = os.path.join(CFDInsight_outputPath, IntersectionPlots_settings.output_name)
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    # Create plot titles
    plot_titles: List[str] = []
    for tpl in IntersectionPlots_settings.field_tuples:
        plot_titles.append("_".join(field.title for field in tpl))


    ### Create single-block domain
    domain_multiBlock, _ = domain_setup.domain_setup(CaseType_settings=CaseType_settings,
                                                     Function_settings=IntersectionPlots_settings)
    
    domain = MergeBlocks(Input=domain_multiBlock)  # type: ignore

    
    ### Iterate through all views, generating images from all of them
    for selected_plot in IntersectionPlots_settings.plots:

        # Create folders to store results
        for plot_title in plot_titles:
            os.makedirs(os.path.join(output_path, selected_plot.title, "plots", plot_title), 
                        exist_ok=True)
            
        # Create dirs for raw and refined csv data
        os.makedirs(os.path.join(output_path, selected_plot.title, "raw_csv_data"), exist_ok=True)
        os.makedirs(os.path.join(output_path, selected_plot.title, "refined_csv_data"), exist_ok=True)

        # Create View objects from settings
        selected_plot.build(domain=domain,
                            caseName=caseName, 
                            modelConfig_summary=modelConfig_summary,
                            plot_titles=plot_titles,
                            field_props_tuples=IntersectionPlots_settings.field_tuples)

        # Generate all screenshots
        selected_plot.create_all_plots(output_path=output_path)

        # Delete object after use
        del selected_plot    