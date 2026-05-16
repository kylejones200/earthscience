"""
Data loaders for geochemistry examples.

Loads and processes Alaska Geochemical Database (AGDB4) data.
"""

import warnings
from pathlib import Path

import pandas as pd


def get_agdb_path() -> Path:
    """Get path to AGDB4 data directory."""
    # Try multiple potential locations
    possible_paths = [
        Path(__file__).parent.parent.parent / "data" / "AGDB4_text",
        Path("data") / "AGDB4_text",
        Path("../data") / "AGDB4_text",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "AGDB4_text directory not found. Please ensure data is in " "'data/AGDB4_text/' directory"
    )


def load_agdb_geology() -> pd.DataFrame:
    """
    Load geological sample information (deduplicated).

    Returns
    -------
    pd.DataFrame
        Geological information with coordinates, sample types, etc.

    Examples
    --------
    >>> geol = load_agdb_geology()
    >>> print(geol.columns)
    >>> print(f"Total samples: {len(geol)}")
    """
    data_path = get_agdb_path()
    geol_file = data_path / "Geol_DeDuped.txt"

    # Load with appropriate dtypes and encoding
    df = pd.read_csv(geol_file, low_memory=False, encoding="latin-1")  # Try latin-1 encoding

    # Convert coordinates to numeric
    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")

    # Remove samples without coordinates
    df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

    return df


def load_agdb_chemistry(elements: list[str] | None = None) -> pd.DataFrame:
    """
    Load geochemical data for specified elements.

    Parameters
    ----------
    elements : list of str, optional
        List of element symbols (e.g., ['Cu', 'Au', 'Ag']).
        If None, loads major elements (SiO2, Al2O3, etc.)

    Returns
    -------
    pd.DataFrame
        Chemical data with AGDB_ID, element, value, units, etc.

    Examples
    --------
    >>> # Load copper and gold data
    >>> chem = load_agdb_chemistry(['Cu', 'Au'])
    >>>
    >>> # Load major elements
    >>> majors = load_agdb_chemistry()
    """
    data_path = get_agdb_path()

    # Determine which files to load based on elements
    chem_files = {
        "Chem_A_Br.txt": ["Ag", "Al", "As", "Au", "B", "Ba", "Be", "Bi", "Br"],
        "Chem_C_Gd.txt": [
            "C",
            "Ca",
            "Cd",
            "Ce",
            "Cl",
            "Co",
            "Cr",
            "Cs",
            "Cu",
            "Dy",
            "Er",
            "Eu",
            "F",
            "Fe",
            "Ga",
            "Gd",
        ],
        "Chem_Ge_Os.txt": [
            "Ge",
            "Hf",
            "Hg",
            "Ho",
            "I",
            "In",
            "Ir",
            "K",
            "La",
            "Li",
            "Lu",
            "Mg",
            "Mn",
            "Mo",
            "N",
            "Na",
            "Nb",
            "Nd",
            "Ni",
            "O",
            "Os",
        ],
        "Chem_P_Te.txt": [
            "P",
            "Pb",
            "Pd",
            "Pr",
            "Pt",
            "Rb",
            "Re",
            "Rh",
            "Ru",
            "S",
            "Sb",
            "Sc",
            "Se",
            "Si",
            "Sm",
            "Sn",
            "Sr",
            "Ta",
            "Tb",
            "Te",
        ],
        "Chem_Th_Zr.txt": ["Th", "Ti", "Tl", "Tm", "U", "V", "W", "Y", "Yb", "Zn", "Zr"],
    }

    # If no elements specified, use common major elements
    if elements is None:
        elements = ["SiO2", "Al2O3", "Fe2O3", "MgO", "CaO", "Na2O", "K2O", "TiO2", "MnO", "P2O5"]

    # Determine which files to load
    files_to_load = []
    if any("O" in elem or "2O3" in elem for elem in elements):
        # Loading oxides - these are in special columns
        files_to_load = list(chem_files.keys())
    else:
        for file, file_elements in chem_files.items():
            if any(elem in file_elements for elem in elements):
                files_to_load.append(file)

    # Load data
    dfs = []
    for filename in files_to_load:
        filepath = data_path / filename
        df = pd.read_csv(filepath, low_memory=False, encoding="latin-1")

        # Filter for requested elements if specified
        if elements:
            # Match by SPECIES column
            df = df[df["SPECIES"].isin(elements)]

        dfs.append(df)

    if not dfs:
        warnings.warn(f"No data found for elements: {elements}")
        return pd.DataFrame()

    # Combine all data
    chem_df = pd.concat(dfs, ignore_index=True)

    # Convert data value to numeric
    chem_df["DATA_VALUE"] = pd.to_numeric(chem_df["DATA_VALUE"], errors="coerce")

    return chem_df


def prepare_spatial_data(elements: list[str], min_samples: int = 10) -> pd.DataFrame:
    """
    Prepare geochemical data for spatial analysis.

    Combines chemistry and geology, pivots to wide format.

    Parameters
    ----------
    elements : list of str
        Elements to include
    min_samples : int
        Minimum number of samples required per element

    Returns
    -------
    pd.DataFrame
        Spatial data with columns: LATITUDE, LONGITUDE, element1, element2, ...

    Examples
    --------
    >>> spatial = prepare_spatial_data(['Cu', 'Au', 'Ag'])
    >>> print(spatial.head())
    """
    # Load data
    geol = load_agdb_geology()
    chem = load_agdb_chemistry(elements)

    # Pivot chemistry to wide format
    chem_wide = chem.pivot_table(
        index="AGDB_ID",
        columns="SPECIES",
        values="DATA_VALUE",
        aggfunc="mean",  # Average if multiple analyses
    )

    # Merge with geology
    spatial_df = geol[
        ["AGDB_ID", "LATITUDE", "LONGITUDE", "PRIMARY_CLASS", "SAMPLE_SOURCE", "QUAD"]
    ].merge(chem_wide, left_on="AGDB_ID", right_index=True, how="inner")

    # Remove samples with too many missing values
    n_elements = len(elements)
    spatial_df = spatial_df.dropna(
        subset=elements, thresh=max(1, int(n_elements * 0.5))  # Require at least 50% of elements
    )

    # Check if we have enough samples
    for elem in elements:
        n_valid = spatial_df[elem].notna().sum()
        if n_valid < min_samples:
            warnings.warn(f"Element {elem} has only {n_valid} samples (< {min_samples})")

    return spatial_df


def load_stream_sediments(elements: list[str]) -> pd.DataFrame:
    """
    Load stream sediment samples only.

    Parameters
    ----------
    elements : list of str
        Elements to load

    Returns
    -------
    pd.DataFrame
        Stream sediment data

    Examples
    --------
    >>> streams = load_stream_sediments(['Cu', 'Zn', 'Pb'])
    """
    spatial = prepare_spatial_data(elements)

    # Filter for stream sediments
    stream_mask = (spatial["SAMPLE_SOURCE"].str.contains("stream", case=False, na=False)) | (
        spatial["PRIMARY_CLASS"].str.contains("sediment", case=False, na=False)
    )

    return spatial[stream_mask].copy()


def load_rock_samples(elements: list[str]) -> pd.DataFrame:
    """
    Load rock samples only.

    Parameters
    ----------
    elements : list of str
        Elements to load

    Returns
    -------
    pd.DataFrame
        Rock sample data

    Examples
    --------
    >>> rocks = load_rock_samples(['SiO2', 'Al2O3', 'Fe2O3'])
    """
    spatial = prepare_spatial_data(elements)

    # Filter for rocks
    rock_mask = (spatial["PRIMARY_CLASS"].str.contains("rock", case=False, na=False)) | (
        spatial["PRIMARY_CLASS"].str.contains(
            "igneous|metamorphic|sedimentary", case=False, na=False
        )
    )

    return spatial[rock_mask].copy()


def get_element_stats(element: str) -> dict:
    """
    Get summary statistics for an element.

    Parameters
    ----------
    element : str
        Element symbol

    Returns
    -------
    dict
        Statistics including count, mean, median, percentiles, etc.

    Examples
    --------
    >>> stats = get_element_stats('Cu')
    >>> print(f"Median Cu: {stats['median']} {stats['units']}")
    """
    chem = load_agdb_chemistry([element])

    if chem.empty:
        return {"error": f"No data found for {element}"}

    values = chem["DATA_VALUE"].dropna()
    units = chem["UNITS"].mode()[0] if "UNITS" in chem.columns else "unknown"

    return {
        "element": element,
        "count": len(values),
        "mean": values.mean(),
        "median": values.median(),
        "std": values.std(),
        "min": values.min(),
        "max": values.max(),
        "p25": values.quantile(0.25),
        "p75": values.quantile(0.75),
        "p95": values.quantile(0.95),
        "units": units,
    }
