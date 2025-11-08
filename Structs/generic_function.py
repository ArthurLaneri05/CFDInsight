from dataclasses import dataclass

""" 
Parent Class for all function-related structures
"""
@dataclass
class Function_Settings:
    """Base class for all function settings with output folder management."""
    run: bool

    def __set_type__(self):
        """String identifier for the output dict name"""
        self.output_name: str = None
    