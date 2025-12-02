"""
Utility functions for handling variable-specific operations such as unit conversions,
cropping to regions, and retrieving plotting specifications.
"""

import logging

# Setup logging
logger = logging.getLogger(__name__)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# --- 1) Sequential colormap: white → teal → slategray ---
precip_colors = [
    
    "#ffffff",  # white (0 mm baseline, not underflow)
    "#c9f3df",  # 
    "#66c29a",  # 
    "#2b8c67",  # 
    "#26534A"   # 
    # "#ffffff",  # white (0 mm baseline, not underflow)
    # "#c9f3eb",  # very light aqua
    # "#66c2a4",  # medium teal
    # "#2b8c85",  # dark teal
    # "#264653"   # slate gray-blue (muted)
]
precip_cmap = LinearSegmentedColormap.from_list("precip_white_tealgray", precip_colors, N=256)

# Set default underflow color for near-zero/negative rainfall
precip_cmap.set_under("#c2c2c2")  # gray for dry pixels (< 0.01 mm)

# --- 2) Diverging colormap: brown → white → teal-gray ---
bias_colors = [
    "#8c510a",  # muted brown
    "#ffffff",  # white center (zero bias)
    "#2b8c85"   # teal-gray for positive bias
]
bias_cmap = LinearSegmentedColormap.from_list("bias_brown_white_tealgray", bias_colors, N=256)

mpl.colormaps.register(cmap=bias_cmap) # type: ignore
mpl.colormaps.register(cmap=precip_cmap) # type: ignore

def get_units(cfg):
    """
        Get the specifications for plotting samples during training.
        Colors, labels, and other parameters are based on the configuration.
    """

    
    units = {"temp": r"$^\circ$C",
             "prcp": "mm",
             "cape": "J/kg",
             "nwvf": "m/s",
             "ewvf": "m/s",
             "msl": "hPa",
             "z_pl_250": "m",
             "z_pl_500": "m",
             "z_pl_850": "m",
             "z_pl_1000": "m",
             }


    hr_unit = units[cfg['highres']['variable']]
    lr_units = []
    for key in cfg['lowres']['condition_variables']:
        if key not in units:
            raise ValueError(f"Variable '{key}' not found in units dictionary.")
        else:
            lr_units.append(units[key])

    return hr_unit, lr_units

def get_unit_for_variable(variable: str):
    """
    Get the unit string for a specific variable.
    """
    units = {
        "temp": r"$^\circ$C",
        "prcp": "mm",
        "cape": "J/kg",
        "nwvf": "m/s",
        "ewvf": "m/s",
        "msl": "hPa",
        "z_pl_250": "m",
        "z_pl_500": "m",
        "z_pl_850": "m",
        "z_pl_1000": "m",
    }

    if variable not in units:
        raise ValueError(f"[get_unit_for_variable] Variable '{variable}' not found in units dictionary.")
    return units[variable]

def correct_variable_units(var_name, model, data):
    
    """
    Apply basic unit corrections to known variables.
    E.g., convert temperature from K to C, precipitation from m to mm.
    """
    if var_name in ["temp", "t2m"]:
        data = data - 273.15
    elif var_name in ["prcp", "tp"] and model in ["DANRA"]:
        # Make sure no negative values (set <0 to 1e-10)
        data[data < 0] = 1e-10
    elif var_name in ["prcp"] and model in ["ERA5"]:
        data = data * 1000  # from m to mm
        # Make sure no negative values after conversion (set <0 to 1e-10)
        data[data < 0] = 1e-10
    elif var_name in ["cape"] and model in ["ERA5"]:
        data = data / 1000  # from J/kg to kJ/kg
        # Also ensure no negative CAPE values
        data[data < 0] = 1e-10
    elif var_name in ["msl"] and model in ["ERA5"]:
        data = data / 100  # from Pa to hPa
    elif var_name in ["pev"] and model in ["ERA5"]:
        data = data / 1000  # from Pa to hPa
    elif var_name in ["z_pl_1000", "z_pl_250", "z_pl_500", "z_pl_850"] and model in ["ERA5"]:
        data = data / 9.81  # from geopotential to geopotential height in meters
        
    return data

def crop_to_region(data, crop_region):
    """
    Crop the data to a specific subregion: [x_start, x_end, y_start, y_end].
    """
    [x_start, x_end, y_start, y_end] = crop_region
    return data[x_start:x_end, y_start:y_end]

def get_var_name_short(varname, model, domain_size=[589, 789]):
    """
    Optionally standardize variable naming (e.g., aliasing or shortening).
    """
    domain_size_str = f"{domain_size[0]}x{domain_size[1]}"

    if model == 'DANRA':
        aliases = {
            "temp": "t2m_ave",
            "prcp": "tp_tot"
        }
    elif model == 'ERA5':
        aliases = {
            "cape": f"cape_{domain_size_str}",
            "ewvf": f"wvf_east_{domain_size_str}",
            "msl": f"msl_{domain_size_str}",
            "nwvf": f"wvf_north_{domain_size_str}",
            "pev": f"pev_{domain_size_str}",
            "prcp": f"tp_{domain_size_str}",
            "temp": f"t2m_{domain_size_str}",
            "z_pl_1000": f"z_pl_1000_hPa_{domain_size_str}",
            "z_pl_250": f"z_pl_250_hPa_{domain_size_str}",
            "z_pl_500": f"z_pl_500_hPa_{domain_size_str}",
            "z_pl_850": f"z_pl_850_hPa_{domain_size_str}"
        }
    else:
        aliases = {}
    return aliases.get(varname, varname)


def get_color_for_variable(variable: str, model: str):
    """
    Get a specific color for a variable based on the model type.
    Models can be DANRA or ERA5 - same variables have different colors for the two models.
    """
    if model.lower() == "danra":
        colors = {
            "temp": "cornflowerblue",
            "prcp": "darkorange",
            "cape": "forestgreen",
            "nwvf": "firebrick",
            "ewvf": "darkmagenta",
            "msl": "teal",
            "z_pl_250": "pink",
            "z_pl_500": "chocolate",
            "z_pl_850": "orange", 
            "z_pl_1000": "royalblue"
        }
    elif model.lower() == "era5":
        colors = {
            "temp": "mediumturquoise",
            "prcp": "goldenrod",
            "cape": "olive",
            "nwvf": "coral",
            "ewvf": "mediumpurple",
            "msl": "skyblue",
            "z_pl_250": "orchid",
            "z_pl_500": "coral",
            "z_pl_850": "tan",
            "z_pl_1000": "midnightblue"
        }
    else:
        # Default colors if model is unknown
        colors = {
            "temp": "cornflowerblue",
            "prcp": "darkorange",
            "cape": "forestgreen",
            "nwvf": "firebrick",
            "ewvf": "darkmagenta",
            "msl": "teal",
            "z_pl_250": "pink",
            "z_pl_500": "chocolate",
            "z_pl_850": "orange", 
            "z_pl_1000": "royalblue"
        }

    if variable not in colors:
        raise ValueError(f"[get_color_for_variable] Variable '{variable}' not found in color dictionary for model '{model}'.")
    
    return colors[variable]


def get_cmaps(cfg):
    """
        Get the colormaps for plotting samples during training.
        Colormaps are based on the configuration.
    """
    cmaps = {"temp": "plasma",
             # set the custom precip colormap
             "prcp": "precip_white_tealgray",
             "cape": "viridis",
             "nwvf": "cividis",
             "ewvf": "magma",
             "msl": "coolwarm",
             "z_pl_250": "coolwarm",
             "z_pl_500": "coolwarm",
             "z_pl_850": "coolwarm",
             "z_pl_1000": "coolwarm",
             }
    

    hr_cmap = cmaps[cfg['highres']['variable']]
    lr_cmaps = {}
    for key in cfg['lowres']['condition_variables']:
        if key not in cmaps:
            raise ValueError(f"Variable '{key}' not found in cmap dictionary.")
        else:
            lr_cmaps[key] = cmaps[key]

    return hr_cmap, lr_cmaps

def get_cmap_for_variable(variable: str):
    """
    Get the matplotlib colormap name for a specific variable.
    """
    cmaps = {"temp": "plasma",
             "prcp": "precip_white_tealgray",
             "prcp_bias": "bias_brown_white_tealgray",
             "cape": "viridis",
             "nwvf": "cividis",
             "ewvf": "magma",
             "msl": "coolwarm",
             "z_pl_250": "coolwarm",
             "z_pl_500": "coolwarm",
             "z_pl_850": "coolwarm",
             "z_pl_1000": "coolwarm",
             }

    if variable not in cmaps:
        # If variable not found, return a default colormap
        logger.warning(f"[get_cmap_for_variable] Variable '{variable}' not found in cmap dictionary. Using default 'viridis'.")
        return "viridis"
    return cmaps[variable]

def get_color_for_model(model_str: str):
    """
    Get a specific color for a model.
    HR/DANRA
    PMM/gen/model/generated
    LR/ERA5
    """
    model_str_norm = model_str.lower()

    # if model_str_norm in ["hr", "danra"]:
    #     return "#a559aa" # Purple
    # elif model_str_norm in ["pmm"]:
    #     return "#0c3d79" # Dark Blue
    # elif model_str_norm in ["lr", "era5"]:
    #     return "#f3bd51" # Gold
    # elif model_str_norm in ["ens", "ensemble", "gen", "generated", "model"]:
    #     return "#439f91" # Teal
    # elif model_str_norm in ["qm", "pmm_qm", "lr_qm", "era5_qm"]:
    #     return "#e76f51" # Coral
    # elif model_str_norm in ["unet", "unet_sr", "unet_gen", "unet_pmm"]:
    #     return "#0e9c26" # Greenish
    # else:
    #     # Default color
    #     return "#7570b3"
    if model_str_norm in ["hr", "danra"]:
        return "#4b4b4b" # Purple
    elif model_str_norm in ["pmm"]:
        return "#288C7D" # Dark Blue
    elif model_str_norm in ["lr", "era5"]:
        return "#997938" # Gold
    elif model_str_norm in ["ens", "ensemble", "gen", "generated", "model"]:
        return "#35B19F" # Teal
    elif model_str_norm in ["qm", "pmm_qm", "lr_qm", "era5_qm"]:
        return "#ab513b" # Coral
    elif model_str_norm in ["unet", "unet_sr", "unet_gen", "unet_pmm"]:
        return "#0a4714" # Greenish
    else:
        # Default color
        return "#7570b3"



# === Model color helpers ===
def get_color_for_model_cycle(models: list[str]) -> list[str]:
    """
    Convenience: return a list of hex colors for a given ordered list of model labels.
    Labels can be e.g. ['HR','PMM','LR'] or ['DANRA','generated','ERA5'].
    Falls back to default color for unknown labels.
    """
    if models is None:
        return []
    return [get_color_for_model(m) for m in models]


def normalize_model_label(label: str) -> str:
    """
    Normalize a legend label to a canonical model keyword used by get_color_for_model.
    Examples:
      'HR (DANRA)' -> 'HR'
      'High-Res'   -> 'HR'
      'PMM (ens)'  -> 'PMM'
      'Gen'|'Generated'|'Model' -> 'generated'
      'LR (ERA5)'  -> 'LR'
    """
    s = (label or "").strip().lower()
    # canonical groups
    if any(k in s for k in ["hr", "danra", "high-res", "high resolution", "truth"]):
        return "hr"
    if any(k in s for k in ["pmm"]):
        return "pmm"
    if any(k in s for k in ["gen", "generated", "model", "ensemble"]):
        return "generated"
    if any(k in s for k in ["lr", "era5", "low-res", "low resolution"]):
        return "lr"
    return s    