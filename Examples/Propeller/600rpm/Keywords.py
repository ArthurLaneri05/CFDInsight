# CFDInsight©, by Arthur Laneri, 2025.
# License: AGPL-3.0-or-later


from typing import Tuple, Dict

"""Case-level keywords for CFDInsight.

This file declares *which* variables the post-processing expects to find
in the `initialConditions` file in the case directory.

This dictionary acts as a **bridge** between:

1. the `initialConditions` file in the case directory  
   (e.g. it may contain a line like:
       magU                 2;    // In m/s
   which is the value **actually** used in the simulation),
2. and the **behaviour** of CFDInsight during post-processing.

How it works
------------
- Each key you put here **must exist** in `initialConditions`. If it doesn’t,
  CFDInsight will raise an error because it assumes that whatever you declare
  here is a real CFD input.
- Once a key is declared here, its value from `initialConditions` becomes
  available inside `Preferences.py` so that you can **sync** parts of the
  post-processing pipeline with the actual CFD inputs.

Typical uses
------------
- If you declare `magU` here, CFDInsight can:
  - read the freestream velocity from `initialConditions`,
  - use it to set field scales (min/max of velocity plots),
  - use it to build expressions, e.g:
        Cp = pMean / (0.5 * magU**2)
- Variables tagged with `show_in_summary=True` will be displayed in the
  configuration summary at the top of all outputs (with unit) to give context.

Returned structure
------------------
{
    "<keyword>": ("<unit>", <show_in_summary: bool>)
}
"""

def get_keywords() -> Dict[str, Tuple[str, bool]]:
    # Follow this scheme:
    #   "key": ("unit", display_in_summary_flag)
    
    keywords = {
        # Used to sync freestream-related plots and derived fields
        "magU": ("m/s", True),

        # Rotor/propeller speed, useful to show in summaries
        "omega": ("rpm", True),

        # Backend CFD parameter: read it, but don’t show it by default
        "rho": ("kg/m3", False),
    }

    return keywords

