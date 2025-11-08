# CFDInsight©, by Arthur Laneri, 2025.
# License: AGPL-3.0-or-later


from typing import Tuple, Dict

"""Case-level keywords for CFDInsight.

This file declares *which* variables the post-processing expects to find
in the `initialConditions` file in the case directory.

Returned structure
------------------
{
    "<keyword>": ("<unit>", <show_in_summary: bool>)
}
"""

def get_keywords() -> Dict[str, Tuple[str, bool]]:
    
    keywords = {}

    return keywords

