from Utils.shared_imports import *
from Structs.generic_function import Function_Settings
from Structs.case_types import *

from Utils.vector_manipulation_utils import get_WindToBody_RotMat, gram_schmidt_ortogonalization
import matplotlib.pyplot as plt


""" Utilities """
def _write_averages_file(
        caseName: str,
        modelConfig_summary: str,
        patchObjs: List["_GenericPatch"],
        A_ref: float,
        c_ref: float,
        b_ref: float,
        CofR: np.ndarray,
        moment_coeff_convention: str,
        coefficient_averaging_iters: int,
        coefficient_decimal_digits: int
    ) -> None:
    
    # Write Geometry Data into string 
    geomVars_str = (f"\n{'A_ref':<16}= {A_ref:.3f}m^2"
                    f"\n{'c_ref':<16}= {c_ref:.3f}m"
                    f"\n{'b_ref':<16}= {b_ref:.3f}m"
                    f"\n{'CofR' :<16}= {np.array2string(CofR, precision=3)}m\n")
    
    # Write Averaging settings into string 
    averaging_settings_str = (
        f"\n{'Moments expr. in':<16}  '{moment_coeff_convention}'"
        f"\n{'Avg. iters.':<16}= {coefficient_averaging_iters}"
        f"\n{'Decimal digits':<16}= {coefficient_decimal_digits}")
                    
    # Write the file
    with open(os.path.join(output_path, "averages.txt"), 'w') as file:

        file.write("="*60 + "\n")
        file.write(f"Case Name: {caseName}".center(60) + "\n")
        file.write("="*60 + "\n")

        file.write("Model attitude and configuration: ")
        file.write(modelConfig_summary)
        file.write("\n \n")

        file.write("Model default geometric parameters: ")
        file.write(geomVars_str)
        file.write("\n \n")

        file.write("Coefficients derivation parameters: ")
        file.write(averaging_settings_str)
        file.write("\n \n")

        file.write("="*60 + "\n")
        file.write(f"Patch-Specific Coefficients:".center(60) + "\n")
        file.write("="*60 + "\n")

        for obj in patchObjs:
            file.write(str(obj) + 2*"\n")

def _read_sim_residuals(residuals_to_plot: List[str]) -> Dict[str, List[float]]:
    """
    Handles aerodynamic residuals data processing for a case folder.

    Args:
        case_folder_path (str): Path to the OpenFOAM case folder.
        residuals_to_plot (List[str]): List of residual names to extract (must
            match the column names in solverInfo.dat).

    Returns:
        Dict[str, List[float]]: Mapping residual name -> list of values.

    Raises:
        FileNotFoundError: If the residuals file doesn't exist.
        RuntimeError: For any other parsing/processing error.
    """
    # Initialize output structure
    raw_data: Dict[str, List[float]] = {k: [] for k in residuals_to_plot}

    # Reads and parses the coefficient data file
    dat_file_path = os.path.join(
        case_folder_path, "postProcessing", "residuals", "0", "solverInfo.dat"
    )

    try:
        with open(dat_file_path, "r") as f:
            # Skip first line
            next(f)

            headers = next(f).strip().lstrip("#").split()
            # Build column index list for the requested residuals
            col_indices = [headers.index(k) for k in raw_data.keys()]

            # Parse each line and append values to the corresponding series
            for line in f:
                values = line.strip().split()
                for k, idx in zip(raw_data.keys(), col_indices):
                    val = float(values[idx])
                    raw_data[k].append(val)

    except FileNotFoundError:
        raise FileNotFoundError(f"Residuals file not found: {dat_file_path}")
    except Exception as e:
        raise RuntimeError(f"Error processing {dat_file_path}: {str(e)}")

    return raw_data

def _parse_dat_file(patch_name: str) -> Tuple[np.ndarray, np.ndarray]:

    # Create temporary dicts
    forces_raw = []
    moments_raw = []

    # Derive paths to .dat files
    base_path = os.path.join(case_folder_path, "postProcessing", 
                             patch_name + "_forces", "0")
    forces_dat_path = os.path.join(base_path, "force.dat")
    moments_dat_path = os.path.join(base_path, "moment.dat")

    # Check .dat files existance
    if not (os.path.isfile(forces_dat_path) and
        os.path.isfile(moments_dat_path)):
        raise FileNotFoundError("""Either moments or forces associated with "
                                    the patch couldn't be found""")

    # Fill the temporary dicts
    with open(forces_dat_path, 'r') as f:
        # Skip first 4 lines
        for _ in range(4):
            next(f)     

        # Go through all the lines
        for line in f:
            values = line.strip().split()
            total_force = values[1:4]
            forces_raw.append(np.array(total_force, dtype=float))

    with open(moments_dat_path, 'r') as f:
        # Skip first 4 lines
        for _ in range(4):
            next(f)  

        # Go through all the lines
        for line in f:
            values = line.strip().split()
            total_moment = values[1:4]
            moments_raw.append(np.array(total_moment, dtype=float))

    # Finish them
    forces_raw = np.array(forces_raw)
    moments_raw = np.array(moments_raw)

    return forces_raw, moments_raw

def _translate_moments(moments_array: np.ndarray, 
                       forces_array: np.ndarray, 
                       pole:np.ndarray) -> np.ndarray:

    return moments_array + np.cross(pole, forces_array)

def _average_data(raw_data_dict: Dict[str, np.ndarray],
                  averaging_iters: int,
                  results_decimal_digits: int
                  ) -> Tuple[Dict[str, np.ndarray], 
                             Dict[str, np.ndarray]]:
    
    avg_data_dict: Dict[str, np.ndarray] = {}
    rms_data_dict: Dict[str, np.ndarray] = {}
    
    for k in raw_data_dict.keys():
        # Extract the last values of the coefficient
        try:
            last_N_vals = raw_data_dict[k][-averaging_iters:]
        except IndexError:
            last_N_vals = raw_data_dict[k]

        # Calculate average and round it to the desired decimal digits
        coeff_avg = np.mean(last_N_vals)
        avg_data_dict.update({k : round(coeff_avg, 
                                        results_decimal_digits)
                            })

        # Calculate rms and round it to the desired decimal digits
        diffs = last_N_vals - avg_data_dict[k]
        coeff_rms = np.sqrt(np.mean(diffs**2))
        rms_data_dict.update({k : round(coeff_rms, 
                                        results_decimal_digits)
                            })

    return avg_data_dict, rms_data_dict

def _create_plot(caseName: str,
                 raw_data_dict: Dict[str, np.ndarray],
                 coefficients_to_plot: List[str],
                 patch_name: Optional[str] = None,
                 patch_type: Optional[str] = None,
                 residuals_plot: bool = False) -> None:
    """Plot selected series vs iteration and save as PNG."""

    # Figure & axes
    fig, ax = plt.subplots(figsize=(9, 6))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # full-bleed canvas

    maxIter = len(next(iter(raw_data_dict.values()))) 
    iters_array = np.arange(1, maxIter+1, 1)

    # Axes styling
    ax.tick_params(axis='both',
                   labelsize=tick_font_size,
                   pad=1,
                   width=1)
    ax.grid(True, alpha=0.2)


    # Plot lines
    for k in coefficients_to_plot:
        ax.plot(iters_array, raw_data_dict[k], label=k)

    # Labels, legend, scale, title
    ax.set_xlabel("Iterations", fontsize=label_font_size)

    if residuals_plot:
        # Set log scale for residuals
        ax.set_yscale("log")
        ax.set_ylabel("Residual Value", fontsize=label_font_size)
        title = r"$\bf{Case Name:}$" + f" {caseName}$\\bf{{;}}$   " + r"$\bf{Residuals}:$"
        out_name = "residuals.png"
    else:
        ax.set_ylabel("Coeff. Value", fontsize=label_font_size)
        ptype = f" [{patch_type}]"
        title = r"$\bf{Case Name:}$" + f" {caseName}$\\bf{{;}}$   "+ r"$\bf{Patch}:$ " + patch_name + ptype
        out_name = f"{patch_name}_Coeffs.png"

    ax.legend(fontsize=label_font_size)
    ax.set_title(title,
                 linespacing=title_linespacing,
                 fontsize=title_font_size,
                 loc='left')

    ### Save figure and close
    plt.savefig(os.path.join(output_path, out_name), 
                dpi=img_dpi, 
                bbox_inches="tight",
                pad_inches=pad_in)
    plt.close()


""" Settings for Preferences.py file """
@dataclass
class Coefficients_Settings(Function_Settings):
    """
    Settings for aerodynamic coefficient calculations and monitoring.

    Attributes:
        run (bool):
            Flag that determines whether the function has to be run.

        A_ref_default (float):
            Reference area in m² used for coefficient calculation (> 0).

        c_ref_default (float):
            Reference chord length in m used in *Cm* calculation (> 0).

            *Note: each patch can override this value, otherwise this default is used.*

        b_ref_default (float):
            Reference span in m used in *Cl*, *Cn* calculation (> 0).

            *Note: each patch can override this value, otherwise this default is used.*
        
        CofR_default (List[float]):
            Reference point [x, y, z] in **meters** used for moment coefficients calculation.

            *Note: each patch can override this value, otherwise this default is used.*
        
        rho_inf (float):
            Freestream density in kg/m³ (> 0).

        U_inf_mag (float):
            Freestream speed magnitude in m/s (> 0).

        moment_coefficients_convention (Literal['wind_axes', 'body_axes']):
            Reference axis system used to express moment coefficients in *BodyPatch* and *SplitBodyPatch* objects.
            * **wind_axes:** moments given in wind axes *(X, Y, Z axes of the **domain ref. frame**)*.
            * **body_axes:** moments given in aircraft body axes *(+X **forward**, +Y **right**, +Z **down**)*.
                Works **only** if `alpha` and `beta` keywords are present in **CaseType_Settings**.

        coefficients_decimal_digits (int):
            Number of decimal places used when displaying coefficients (> 0).

        coefficients_avg_iters (int):
            Number of most recent iterations over which running averages of
            coefficients are computed (> 0).

        complete_patches (List["BodyPatch"]):
            Complete (non-split) patches.

        split_patches (List["SplitBodyPatch"]):
            Patches that were split in half due to a symmetric domain.

        rotor_patches (List["RotorPatch"]):
            Rotor/propeller patches.

        residuals_of_interest (List[str]):
            Solver residual field names to monitor (e.g., "Ux_final", "p_final").
    """

    A_ref_default: float
    c_ref_default: float
    b_ref_default: float
    CofR_default: List[float]
    rho_inf: float
    U_inf_mag: float
    moment_coefficients_convention: Literal['wind_axes', 'body_axes']
    coefficients_decimal_digits: int
    coefficients_avg_iters: int
    complete_patches: List["BodyPatch"]
    split_patches: List["SplitBodyPatch"]
    rotor_patches: List["RotorPatch"]
    residuals_of_interest: List[str]
    

    def __post_init__(self):
        """Initialize with default output folder name for coefficients."""
        self.output_name = "Coefficients"

        # Check that rho, A_ref, c_ref, b_ref are > 0
        if self.A_ref_default <= 0:
            raise ValueError("A_ref must be > 0")
        if self.b_ref_default <= 0:
            raise ValueError("b_ref must be > 0")
        if self.c_ref_default <= 0:
            raise ValueError("c_ref must be > 0")
        if len(self.CofR_default) != 3:
            raise ValueError("CofR must have 3 components")
        if self.rho_inf <= 0:
            raise ValueError("rho must be > 0")
        if self.U_inf_mag <= 0:
            raise ValueError("U_inf_mag must be > 0")
        
        # Convert CofR_default to a numpy array
        self.CofR_default = np.array(self.CofR_default)
        
        # Calculate dynamic pressure
        self.dyn_pressure = 0.5 * self.rho_inf * self.U_inf_mag**2
   
        # Create all_patchObjs list for commodity
        self.all_patchObjs = (self.complete_patches + 
                              self.split_patches + 
                              self.rotor_patches)

        # Check that coefficient_decimal_digits 
        # and coefficient_averaging_iters are > 0
        if self.coefficients_decimal_digits <= 0:
            raise ValueError("coefficients_decimal_digits must be > 0")
        if self.coefficients_avg_iters <= 0:
            raise ValueError("coefficient_avg_iters must be > 0")


""" PatchCoeffs classes declaration """
class _GenericPatch(ABC):
    """ Private parent class"""

    def __init__(self,
                 name: str,
                 A_ref: Optional[float] = None,
                 c_ref: Optional[float] = None,
                 b_ref: Optional[float] = None,
                 CofR: Optional[List[float]] = [None, None, None]
                ):
        
        """ For Configuration only """

        # Check that A_ref, c_ref, b_ref are > 0
        if A_ref and A_ref <= 0:
            raise ValueError("A_ref must be > 0")
        if b_ref and b_ref <= 0:
            raise ValueError("b_ref must be > 0")
        if c_ref and c_ref <= 0:
            raise ValueError("c_ref must be > 0")
        if CofR != [None, None, None] and len(CofR) != 3:
            raise ValueError("CofR must have 3 components")
        
        # Attributes initialization
        self.name = name
        self.A_ref = A_ref
        self.c_ref = c_ref
        self.b_ref = b_ref
        self.CofR = CofR

        # If any reference value is different to the dafault,
        # make the __str__ method mention that
        self._non_default_ref_entries = (A_ref, c_ref, b_ref, CofR) != (None, None, None, 
                                                                        [None, None, None])

        # Declare placeholders
        self.type: str = None
        self.coeffs_toPlot: List[str] = None
        self.raw_data: Dict[str, np.ndarray] = None
        self.avg_data: Dict[str, float] = None
        self.rms_data: Dict[str, float] = None

    def build(self,
              Coefficients_settings: Coefficients_Settings,
              CaseType_settings: CaseType_Settings
              ) -> None:
        """ Builds the object from settings """

        ### Set defaults (if needed) for A, b, c and CofR
        if hasattr(self, 'A_ref') and not self.A_ref:
            self.A_ref = Coefficients_settings.A_ref_default
        if hasattr(self, 'b_ref') and not self.b_ref:
            self.b_ref = Coefficients_settings.b_ref_default
        if hasattr(self, 'c_ref') and not self.c_ref:
            self.c_ref = Coefficients_settings.c_ref_default 
        if hasattr(self, 'CofR') and None in self.CofR:
            self.CofR = Coefficients_settings.CofR_default 


        ### Translate CofR into a numpy array
        self.CofR = np.array(self.CofR)


        ### Get raw forces an moments at self.CofR
        forces_raw, moments_raw = _parse_dat_file(self.name)
        moments_raw = _translate_moments(moments_raw, 
                                         forces_raw, 
                                         self.CofR)

        ### Fill in self.raw_data dict
        self.raw_data = self._derive_raw_data(raw_forces=forces_raw,
                                              raw_moments=moments_raw,
                                              Coefficients_settings=Coefficients_settings,
                                              CaseType_settings=CaseType_settings)

        ### Compute and store averages (and respective rms) over the last specified iterations
        self.avg_data, self.rms_data = _average_data(self.raw_data,
                                                     averaging_iters=Coefficients_settings.coefficients_avg_iters,
                                                     results_decimal_digits=Coefficients_settings.coefficients_decimal_digits)  

    def __str__(self) -> str:

        # Add patch name and type
        output = f"{self.name} [{self.type}]:\n"

        # Optionally add non-default reference values
        if self._non_default_ref_entries:
            output += (f"A_ref = {self.A_ref:.3f} m^2\n"
                       f"c_ref = {self.c_ref:.3f} m\n"
                       f"b_ref = {self.b_ref:.3f} m\n"
                       f"CofR = {np.array2string(self.CofR, precision=3)}m\n\n")

        # Print avg. coeffs and corresponding rms
        for k, v in self.avg_data.items():
            output += f"{k:<6}= {v:<10}(rms = {self.rms_data[k]})\n"

        return output        

    @abstractmethod
    def _derive_raw_data(self,
                         raw_forces: np.ndarray,
                         raw_moments: np.ndarray,
                         Coefficients_settings: Coefficients_Settings,
                         CaseType_settings: CaseType_Settings
                         ) -> Dict[str, np.ndarray]:
        pass


class BodyPatch(_GenericPatch):
    """Handles aerodynamic coefficient data processing for a complete surface patch."""

    def __init__(self,
                 name: str,
                 A_ref: Optional[float] = None,
                 c_ref: Optional[float] = None,
                 b_ref: Optional[float] = None,
                 CofR: Optional[List[float]] = [None, None, None]
                ):
        
        """
        Represents a single surface patch for aerodynamic coefficient processing.

        This patch pulls integrated forces/moments from the OpenFOAM forces utility
        named *`{name}_forces`* and normalizes them into coefficients using the
        provided reference metrics. 
        
        The `CofR ();` entry in the OpenFOAM `forces` utility using has to be set up to **`(0 0 0)`**.

        Args:
            name (str):
                Name of the patch to process. The script expects a forces log named
                *`{name}_forces`*.

            A_ref (float):
                Reference area in **m²** used for coefficient normalization (e.g., wing planform area).
                **Note:** if a per-patch value is not specified, the global `A_ref_default` is used.

            c_ref (float):
                Reference chord length in **meters** used for pitching-moment coefficient normalization.
                **Note:** if a per-patch value is not specified, the global `c_ref_default` is used.

            b_ref (float):
                Reference span in **meters** used for rolling and yawing moment coefficient normalization (Cl, Cn).
                **Note:** if a per-patch value is not specified, the global `b_ref_default` is used.

            CofR (List[float]):
                Reference point [x, y, z] in **meters** used for moment coefficients calculation.
                **Note:** if a per-patch value is not specified, the global `CofR_default` is used.
        

        **Outputs:**
            - **CD:**
                Drag coefficient: Computed from the force component along '+x_wind' (**domain** x axis):
                `CD = -F_x_wind / (q · A_ref)`

            - **CS:**
                Side (lateral) force coefficient: Computed from the force component along '+y_wind' (**domain** y axis):
                `CS = F_y_wind / (q · A_ref)`

            - **CL:**
                Lift force coefficient: Computed from the force component along '+z_wind' (**domain** z axis):
                `CL = F_z_wind / (q · A_ref)`

            - **Cl:**
                Rolling moment coefficient (about '+x_wind'/'+x_body', see the notes below for further info):
                'Cl = M_x_body / (q · A_ref · b_ref)'

            - **Cm:**
                Pitching moment coefficient (about '+y_wind'/'+y_body', see the notes below for further info):
                `Cm = M_y_body / (q · A_ref · c_ref)`

            - **Cn:**
                Yawing moment coefficient (about '+z_wind'/'+z_body', see the notes below for further info):
                `Cn = M_z / (q · A_ref · b_ref)`

        **Notes:**
            - **Axes & rotation:** Moment coefficients are expressed in **body axes** only if:
                * `moment_coefficients_convention == 'body_axes'` inside **Coefficients_Settings**.
                * The CaseType is **ExternalAerodynamics**.
                * The incidence angles (α, β) are specified.
        """
        
        super().__init__(name,
                         A_ref=A_ref,
                         b_ref=b_ref,
                         c_ref=c_ref,
                         CofR=CofR)
        self.type = 'BodyPatch'
        self.coeffs_toPlot = ['CD', 'CL', 'Cm']

    def _derive_raw_data(self, 
                         raw_forces: np.ndarray, 
                         raw_moments: np.ndarray, 
                         Coefficients_settings: Coefficients_Settings,
                         CaseType_settings: CaseType_Settings
                         ) -> Dict[str, np.ndarray]:

        ### Decide names of coefficients and moment reference axes
        moments_rotMat: np.ndarray = None
        forceCoeffs_names  = ['CD', 'CS', 'CL']
        momentCoeffs_names = ['Cl', 'Cm', 'Cn']
        if (Coefficients_settings.moment_coefficients_convention == 'body_axes'
            and isinstance(CaseType_settings, ExternalAerodynamics)):

            # Raise error if no alpha/beta value is found
            if (CaseType_settings.alpha is None or
                CaseType_settings.beta  is None):
                raise ValueError("No valid values were found for 'alpha' and 'beta'")
            moments_rotMat = get_WindToBody_RotMat(alpha_deg=CaseType_settings.alpha,
                                                   beta_deg=CaseType_settings.beta)
        else:
            moments_rotMat = np.eye(3)

        ### Create and fill raw_data_dict
        raw_moments = raw_moments @ moments_rotMat
        dyn_pressure = Coefficients_settings.dyn_pressure

        # From forces to coefficients
        forceCoeffs_raw = raw_forces / dyn_pressure / self.A_ref
        momentCoeffs_raw = raw_moments / dyn_pressure / self.A_ref
        momentCoeffs_raw /= np.array([self.b_ref, self.c_ref, self.b_ref])

        raw_data_dict = {
            **{k: np.asarray(forceCoeffs_raw[:, i]).ravel()
            for i, k in enumerate(forceCoeffs_names)},
            **{k: np.asarray(momentCoeffs_raw[:, i]).ravel()
            for i, k in enumerate(momentCoeffs_names)},
        }

        return raw_data_dict
    
    
class SplitBodyPatch(_GenericPatch):
    """Handles aerodynamic coefficient data processing for a surface patch
    that was split in half due to a symmetric domain.
    
    **Note:** an error will be raised if the initialization of a 
    *SplitBodyPatch* is attempted for a non-symmetric case."""

    def __init__(self,
                 name: str,
                 A_ref: Optional[float] = None,
                 c_ref: Optional[float] = None,
                 b_ref: Optional[float] = None,
                 CofR: Optional[List[float]] = [None, None, None]
                 ):
        """
        Represents a surface patch that was **split in half** due 
        to a **symmetric** domain for aerodynamic coefficient processing.

        This patch pulls integrated forces/moments from the OpenFOAM forces utility
        named *`{name}_forces`* and normalizes them into coefficients using the
        provided reference metrics. 
        
        The `CofR ();` entry in the OpenFOAM `forces` utility using has to be set up to **`(0 0 0)`**.

        **Note:** an error will be raised if the initialization of a 
        *SplitBodyPatch* is attempted for a **non-symmetric** case.

        Args:
            name (str):
                Name of the patch to process. The script expects a forces log named
                *`{name}_forces`*.

            A_ref (float):
                Reference area in **m²** used for coefficient normalization (e.g., wing planform area).
                **Note:** if a per-patch value is not specified, the global `A_ref_default` is used.

            c_ref (float):
                Reference chord length in **meters** used for pitching-moment coefficient normalization.
                **Note:** if a per-patch value is not specified, the global `c_ref_default` is used.

            b_ref (float):
                Reference span in **meters** used for rolling and yawing moment coefficient normalization (Cl, Cn).
                **Note:** if a per-patch value is not specified, the global `b_ref_default` is used.

            CofR (List[float]):
                Reference point [x, y, z] in **meters** used for moment coefficients calculation.
                **Note:** if a per-patch value is not specified, the global `CofR_default` is used.

        **Outputs:**
            - **CD:**
                Drag coefficient: Computed from the force component along '+x_wind' (**domain** x axis):
                `CD = -F_x_wind / (q · A_ref)`

            - **CS:**
                Side (lateral) force coefficient: Computed from the force component along '+y_wind' (**domain** y axis):
                `CS = F_y_wind / (q · A_ref)`

            - **CL:**
                Lift force coefficient: Computed from the force component along '+z_wind' (**domain** z axis):
                `CL = F_z_wind / (q · A_ref)`

            - **Cl:**
                Rolling moment coefficient (about '+x_wind'/'+x_body', see the notes below for further info):
                'Cl = M_x_body / (q · A_ref · b_ref)'

            - **Cm:**
                Pitching moment coefficient (about '+y_wind'/'+y_body', see the notes below for further info):
                `Cm = M_y_body / (q · A_ref · c_ref)`

            - **Cn:**
                Yawing moment coefficient (about '+z_wind'/'+z_body', see the notes below for further info):
                `Cn = M_z / (q · A_ref · b_ref)`

        **Notes:**
            - **Axes & rotation:** Moment coefficients are expressed in **body axes** only if:
                * `moment_coefficients_convention == 'body_axes'` inside **Coefficients_Settings**.
                * The CaseType is **ExternalAerodynamics**.
                * The incidence angles (α, β) are specified.
            
        """
        # Init as _Genericpatch
        super().__init__(name=name,
                         A_ref=A_ref,
                         b_ref=b_ref,
                         c_ref=c_ref,
                         CofR=CofR)
        self.type = "SplitBodyPatch"
        self.coeffs_toPlot = ['CD', 'CL', 'Cm']
        
    def build(self,
              Coefficients_settings: Coefficients_Settings,
              CaseType_settings: CaseType_Settings
              ) -> None:
        """ Builds the object from settings """

        # Ensure that the case is 'Symmetric'
        if (not isinstance(CaseType_settings, ExternalAerodynamics)
            or CaseType_settings.variant == 'Asymmetric'):
            raise SyntaxError("A 'SplitBodyPatch' can't be declared for a non-symmetric domain")

        # Build as BodyPatch first
        super().build(Coefficients_settings=Coefficients_settings,
                            CaseType_settings=CaseType_settings)
        
        # Initialize mirroring axis
        self.mirroring_axis = CaseType_settings.mirroring_axis

        # Apply corrections to raw coefficients to account for patch symmetry
        self._apply_symmetry_corrections()

    def _derive_raw_data(self, 
                         raw_forces: np.ndarray, 
                         raw_moments: np.ndarray, 
                         Coefficients_settings: Coefficients_Settings,
                         CaseType_settings: CaseType_Settings
                         ) -> Dict[str, np.ndarray]:
        
        BodyPatch.build(raw_forces=raw_forces,
                        raw_moments=raw_moments,
                        Coefficients_settings=Coefficients_settings,
                        CaseType_settings=CaseType_settings)

    def _apply_symmetry_corrections(self) -> None:
        """Doubles coefficients (the force coefficient associated with the mirroring axis,
        and the moment coefficients associated with the remaining axes are set to 0 instead)"""

        # Use tuples as keys (hashable)
        coeffs_to_zero_dict = {
            (1, 0, 0): ("CD", "Cm", "Cn"), 
            (0, 1, 0): ("CS", "Cl", "Cn"),
            (0, 0, 1): ("CL", "Cl", "Cm")
        }

        # Double all values in-place
        for key in self.raw_data:
            self.raw_data[key] = [x * 2 for x in self.raw_data[key]]

        # Zero selected coefficients
        for coeff_name in coeffs_to_zero_dict[tuple(self.mirroring_axis)]:
            self.raw_data[coeff_name] *= 0


class RotorPatch(_GenericPatch):
    def __init__(self,
                 name: str,
                 origin: List[float],
                 x_axis: List[float],
                 z_axis: List[float],
                 D: float,
                 omega: float
                ):
        
        """
        Handles aerodynamic coefficient data processing for a *rotor/propeller* patch.

        This patch builds a right-handed local rotor frame from the user-supplied
        axes, pulls integrated forces/moments from the OpenFOAM forces utility
        named *`{name}_forces`* and normalizes them into rotor/propeller coefficients.
        
        The `CofR ();` entry in the OpenFOAM `forces` utility using has to be set up to **`(0 0 0)`**.

        Args:
            name (str):
                Identifier for the rotor patch (used in logs/prints).

            origin (List[float]):
                Reference point [x, y, z] in **meters** used for moment coefficients calculation.

            x_axis (List[float]):
                Direction of **rotor drag** in the *local* rotor frame.
                + *The x_axis vector is automatically orthogonalized with z_axis using 'Gram-Schmidt' orthogonalization.*

            z_axis (List[float]):
                Direction of **expected thrust** (positive thrust acts **+z** in
                the rotor frame).

            D (float):
                Rotor/propeller **diameter** in meters (> 0).

            omega (float):
                Rotor speed in **rpm** (positive value means *counter-clockwise
                rotation* when looking along `+z_axis`).

        **Conventions & frames:**
            - **Freestream** is assumed along the **global** +x direction, i.e.
              `U_inf = [U_inf_mag, 0, 0]`.
            - The rotor reference frame is built so that:
              `x_axis` → drag direction, `z_axis` → thrust direction,
              `y_axis = z_axis × x_axis` (right-handed).
            - Forces/moments are projected onto this local frame before
              coefficient normalization.

        **Outputs:**
            - **J:**
            Propeller advance ratio. Computed from the **axial** freestream component along `+z_rotor`:
            `J = U_axial / (n · D)` with `n = omega / 60` (rps) and `D` the rotor diameter.

            - **mu:**
            Rotor advance ratio. Uses the **in-plane** (disk-plane) freestream component:
            `μ = U_in_plane / (Ω · R)` with `Ω = omega / 60 * 2π` (rad/s).

            - **Ct:**
            Thrust coefficient (thrust along `+z_rotor`):
            `Ct = F_z_rotor / (ρ · n^2 · D^4)`.

            - **Cx, Cy:**
            In-plane force coefficients (local rotor frame):
            `Cx = F_x_rotor / (ρ · n^2 · D^4)`, `Cy = F_y_rotor / (ρ · n^2 · D^4)`
            where `x` is the **drag** direction and `y` the **sideforce** direction in the rotor frame.

            - **Cq:**
            Torque coefficient about `+z_rotor`:
            `Cq = Q_z_rotor / (ρ · n^2 · D^5)`.
            Sign convention (required driving torque): `Cq ← -sign(n) · Cq`.

            - **Cp:**
            Power coefficient:
            `Cp = 2π · Cq`.

            - **Eta:**
            Propulsive efficiency:
            `Eta = (T · V∞) / P = (Ct · J) / Cp` (using the **axial** component of `V∞`).

        **External reference data:**
            For further clarification about the conventions used, visit this link:
            https://m-selig.ae.illinois.edu/props/propDB.html
        """

        # Check that rotor_D, is > 0
        if D <= 0:
            raise ValueError("rotor_D must be > 0")     
        if len(origin) != 3 or None in origin:
            raise ValueError("origin must have 3 components")     
           
        
        # Attributes initialization
        self.name = name
        self.CofR = origin
        self.D = D
        self.swept_A = np.pi/4 * D**2
        self.omega = omega/60.0      # Expressed in rps

        # Create ortonormal reference frame
        (self.x_axis, 
         self.y_axis, 
         self.z_axis) = gram_schmidt_ortogonalization(x_axis, z_axis)
        
        # Useful attributes
        self.type = 'RotorPatch'
        self.coeffs_toPlot = ['Ct', 'Cq']
        
        # Declare placeholders
        self.J: float = None
        self.mu: float = None
        self.raw_data: Dict[str, np.ndarray] = None
        self.avg_data: Dict[str, float] = None
        self.rms_data: Dict[str, float] = None

    def __str__(self) -> str:
        output = f"{self.name} [{self.type}]:\n"

        # Origin, D, area, omega
        output += (
            f"Origin = {np.array2string(self.CofR, precision=3)}m\n"
            f"D = {self.D:.3f} m\n"
            f"A = {self.swept_A:.4f} m^2\n"
            f"omega = {self.omega:.3f} rps\n"
        )

        # Ref. frame info
        output += (
            f"Reference frame: x={np.array2string(self.x_axis, precision=3)}; "
            f"y={np.array2string(self.y_axis, precision=3)}; "
            f"z={np.array2string(self.z_axis, precision=3)}\n"
        )
        output += "\n"
        # J, mu
        output += f"J   = {self.J:<10}\n"
        output += f"mu  = {self.mu:<10}\n"

        if self.avg_data is not None and self.rms_data is not None:
            for k, v in self.avg_data.items():
                output += f"{k:<4}= {v:<10}(rms={self.rms_data.get(k, 0.0)})\n"

        return output

    def _derive_raw_data(self, 
                         raw_forces: np.ndarray, 
                         raw_moments: np.ndarray, 
                         Coefficients_settings: Coefficients_Settings,
                         CaseType_settings: CaseType_Settings
                         ) -> Dict[str, np.ndarray]:

        ### Calculate moments and forces components 
        #   in the rotor reference frame
        rotation_matrix = np.vstack((self.x_axis, self.y_axis, self.z_axis))
        raw_forces = raw_forces @ rotation_matrix
        raw_moments = raw_moments @ rotation_matrix

        # Rewrite useful quantities
        U_inf_mag = Coefficients_settings.U_inf_mag
        rho_inf = Coefficients_settings.rho_inf


        ### Derive J and mu and store them inside the object
        # J: Propeller advance ratio
        U_inf_normal_comp = np.dot(np.array([U_inf_mag,0,0]), self.z_axis)
        self.J = np.round(-U_inf_normal_comp / (abs(self.omega) * self.D),
                          Coefficients_settings.coefficients_decimal_digits)

        # mu: Rotor advance ratio
        U_inf_onDisk_comp = np.array([U_inf_mag,0,0]) - U_inf_normal_comp*self.z_axis
        self.mu = np.round(np.linalg.norm(U_inf_onDisk_comp) / (abs(self.omega) * 2*np.pi * self.D/2),
                           Coefficients_settings.coefficients_decimal_digits)


        ### Create and fill raw_data_dict
        raw_data_dict: Dict[str, np.ndarray] = {}

        # Ct: Thrust coefficient
        Ct = raw_forces[:,2] / (rho_inf * self.omega**2 * self.D**4)
        raw_data_dict.update({'Ct' : Ct})

        # Cx: Drag coefficient
        Cx = raw_forces[:,0] / (rho_inf * self.omega**2 * self.D**4)
        raw_data_dict.update({'Cx' : Cx})

        # Cy: Sideforce coefficient
        Cy = raw_forces[:,1] / (rho_inf * self.omega**2 * self.D**4)
        raw_data_dict.update({'Cy' : Cy})

        # Cq: Torque coefficient (The sign is set according to the rotation direction)
        Cq = raw_moments[:,2] / (rho_inf * self.omega**2 * self.D**5)
        Cq = -np.sign(self.omega) * Cq      # *-1 as it expresses the nec. torque to make the rotor turn
        raw_data_dict.update({'Cq' : Cq})

        # Cp: Power coefficient
        Cp = Cq * 2*np.pi
        raw_data_dict.update({'Cp' : Cp})

        # Eta: Rotor efficiency
        eta = Ct * self.J / Cp
        raw_data_dict.update({'Eta' : eta})
        
        return raw_data_dict
        

""" Main function declaration """    
def Main(Coefficients_settings: Coefficients_Settings,
         CaseType_settings: CaseType_Settings,
         caseName: str, 
         modelConfig_summary: str) -> None:
        
    ### Output setup
    global output_path
    output_path = os.path.join(CFDInsight_outputPath, Coefficients_settings.output_name)
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    # Build BodyPatch objects and generate plots
    for patch_obj in Coefficients_settings.all_patchObjs:

        patch_obj.build(CaseType_settings=CaseType_settings,
                        Coefficients_settings=Coefficients_settings)
        
        _create_plot(caseName=caseName,
                     raw_data_dict=patch_obj.raw_data,
                     coefficients_to_plot=patch_obj.coeffs_toPlot,
                     patch_name=patch_obj.name,
                     patch_type=patch_obj.type)

    # Create "averages.txt" file
    _write_averages_file(
        caseName=caseName,
        modelConfig_summary=modelConfig_summary,
        patchObjs=Coefficients_settings.all_patchObjs,
        A_ref=Coefficients_settings.A_ref_default,
        c_ref=Coefficients_settings.c_ref_default,
        b_ref=Coefficients_settings.b_ref_default,
        CofR=Coefficients_settings.CofR_default,
        moment_coeff_convention=Coefficients_settings.moment_coefficients_convention,
        coefficient_averaging_iters=Coefficients_settings.coefficients_avg_iters,
        coefficient_decimal_digits=Coefficients_settings.coefficients_decimal_digits
        )
    
    # Create SimResiduals object and residuals plot (if the residuals values are found)
    raw_residuals = _read_sim_residuals(Coefficients_settings.residuals_of_interest)
    _create_plot(caseName=caseName,
                 raw_data_dict=raw_residuals,
                 coefficients_to_plot=Coefficients_settings.residuals_of_interest,
                 residuals_plot=True)