from Utils.shared_imports import *
import re
from Structs.case_types import CaseType_Settings
from Structs.generic_variable import variable
from Structs.generic_function import Function_Settings



""" Display case settings """
def display_case_information_and_selected_funtions(caseName: str,
                                                   CaseType_settings: CaseType_Settings,
                                                   function_settings_list: List[Function_Settings]) -> None:
    
    ### Initial banner
    print("="*64)
    print("CFDInsight by Arthur Laneri.")

    ### Print Case name:
    print("\n"+ "="*64)
    print(f"CaseName: {caseName}\n")

    ### Print Case type features:
    print(CaseType_settings)

    print("\n"+ "="*64)
    print("Functions to be run:")
    ### Print names of functions to be run:
    for func_settings in function_settings_list:
        if func_settings.run:
            print(f"\t* {func_settings.output_name}")

    print("\n"+ "="*64)


""" Create initialConditions_data dict """
def dict_from_initialConditions(keywords: dict,
                                case_folder: Optional[Union[os.PathLike, str]] = None
                                ) -> Dict[str, variable]:
    """Extract values from the 'initialConditions' file and return them as a dictionary."""

    # local sentinels / patterns
    _missing = np.nan
    _float_line = re.compile(
        r"^\s*(?P<key>[A-Za-z_]\w*)\s+(?P<val>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)[;\s]*$"
    )

    # resolve path
    folder = case_folder if case_folder is not None else case_folder_path
    ic_path = os.path.join(folder, "initialConditions")
    if not os.path.isfile(ic_path):
        raise FileNotFoundError(f"Missing initialConditions file at: {ic_path}")

    # seed output from keywords: {"key": ("unit", show_flag)}
    output: Dict[str, variable] = {}
    for key, meta in keywords.items():
        unit, show_flag = meta
        output[key] = variable(key, _missing, unit, show_flag)

    # parse file (tolerant to whitespace and end-of-line comments)
    with open(ic_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].split("//", 1)[0].strip()
            if not line:
                continue
            m = _float_line.match(line)
            if not m:
                continue
            k = m.group("key")
            if k in output:
                try:
                    output[k].value = float(m.group("val"))
                except ValueError:
                    continue

    # validate presence
    missing = [k for k, var in output.items() if np.isnan(var.value)]
    if missing:
        raise ValueError(f"No acceptable value found for keywords: {', '.join(missing)}")

    return output


""" Get Case Name """
def get_caseName(case_folder: Union[os.PathLike, str] = case_folder_path) -> str:
    """
    Extract a human-readable case name from the folder path.

    Args:
        case_folder : os.PathLike | str, optional
            Path to the case folder. Defaults to `case_folder_path`.

    Returns:
        str
            A formatted case name in the form 'versionName (simTitle)'.
    """    
    
    split_path = case_folder.split('/')
    versionName = split_path[len(split_path)-2]; simTitle = split_path[len(split_path)-1]

    return (versionName + f" ({simTitle})")


""" Create Model Configuration Summary """
def create_summary(initialConditions_data: Dict[str, variable], 
                   characters_per_line: int, 
                   for_coefficients: bool=False) -> str:
    """
    Create a formatted summary string of model configuration.

    Args:
        initialConditions_data : Dict[str, var]
            Dictionary of variables from initial conditions.
        characters_per_line : int
            Maximum number of characters per summary line before wrapping.
        for_coefficients : bool, optional
            If True, create summary formatted for coefficients. 
            Defaults to False.

    Returns:
        str
            Formatted model configuration summary string.
    """
 
    # initialize vars
    modelConfig_summary = ""; curr_line_length = 0

    ### For Coefficients
    if for_coefficients:
        for k, v in initialConditions_data.items():
            if v.show_in_config_summary:
                modelConfig_summary += f"\n{k:<16}= {v.value}{v.unit}"
        
        return modelConfig_summary


    ### For Slices, Surfaces
    for k, v in initialConditions_data.items():
        if v.show_in_config_summary:
            # Add "{Key}: {Value}{Unit};\t" using LateX formatting
            entry = (
                f"$\\mathbf{{{k}:}}$ "       # Bold key
                f"{v.value:.2f}{v.unit}"    # Value + unit
                f"$\\mathbf{{;}}$    "         # Bold semicolon + spaces
                    )
            
            if curr_line_length + len(entry) > characters_per_line:
                curr_line_length = len(entry)
                modelConfig_summary += "\n" + entry; 
            else:
                curr_line_length += len(entry)
                modelConfig_summary += entry; 

    return modelConfig_summary


# """ Get Surface Settings from Baseline """
# def importPreferencesModule(directory):
#     preferences_path = os.path.join(directory, "Preferences.py")
    
#     if not os.path.isfile(preferences_path):
#         print("Preferences.py not found in the specified directory.")
#         exit()
#         return None
    
#     spec = importlib.util.spec_from_file_location("Preferences", preferences_path)
#     preferences_module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(preferences_module)

#     return preferences_module