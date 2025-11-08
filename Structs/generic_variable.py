from dataclasses import dataclass


@dataclass
class variable:
    """
    A generic variable container storing a numerical value with its unit and display preferences.
    
    This class provides a standardized way to handle physical quantities throughout the application,
    ensuring consistent unit handling and configuration reporting.
    """
    _name: str
    """The name of the variable (e.g., "U_mag", "MAC", "alpha")"""
    
    value: float
    """The numerical value of the variable (e.g., 10.5, -3.14, 1000.0)"""
    
    unit: str
    """The physical unit of the value (e.g., 'm/s', 'Pa', 'kg/m³', '°')"""
    
    show_in_config_summary: bool = False
    """
    Flag indicating whether this variable should be included in configuration summaries.
    
    When True, the variable will be displayed in reports and configuration overviews.
    When False, the variable is used internally but not shown in summaries.
    """
    

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the variable.
        
        Returns:
            Formatted string with value and unit (e.g., "10.5m/s")
        """
        name_w_colon = self._name + ":"
        return f"{name_w_colon:<16}{self.value:.2f}{self.unit}"
    
    def summary_entry(self) -> str:
        """
        Generate a formatted entry for configuration summaries.
        
        Returns:
            String suitable for inclusion in reports, or empty string if not shown
        """
        if self.show_in_config_summary:
            return (
                f"$\\mathbf{{{self._name}:}}$ "       # Bold name
                f"{self.value:.2f}{self.unit}"      # Value with 2 decimal places + unit
                f"$\\mathbf{{;}}$"                   # Bold semicolon
                    )
        return ""
    