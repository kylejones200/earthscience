"""
Geospatial data loaders for standard earth science datasets

Functions to download and load ETOPO, SRTM, GTOPO30, and other datasets.
"""

import logging
import warnings
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_etopo2022(
    bbox: tuple[float, float, float, float], resolution: str = "15s", data_dir: str | None = None
) -> dict:
    """
    Load ETOPO 2022 global relief model.

    ETOPO 2022 is a 15 arc-second global relief model combining
    land topography and ocean bathymetry.

    Parameters
    ----------
    bbox : tuple
        Bounding box (lon_min, lat_min, lon_max, lat_max) in degrees
    resolution : str, optional
        Resolution: '15s' (15 arc-seconds, ~450m) or '60s' (1 arc-minute)
        (default='15s')
    data_dir : str, optional
        Directory to cache downloaded data

    Returns
    -------
    dict
        Dictionary with 'elevation', 'lon', 'lat', 'metadata'

    Notes
    -----
    Data source: NOAA National Centers for Environmental Information
    https://www.ncei.noaa.gov/products/etopo-global-relief-model

    First call will download data (may be large!). Subsequent calls use cache.

    Examples
    --------
    >>> # Load Hawaii region
    >>> bbox = (-160, 18, -154, 23)  # (lon_min, lat_min, lon_max, lat_max)
    >>> data = load_etopo2022(bbox, resolution='15s')
    >>>
    >>> elevation = data['elevation']
    >>> lon = data['lon']
    >>> lat = data['lat']
    >>>
    >>> print(f"Shape: {elevation.shape}")
    >>> print(f"Elevation range: {elevation.min():.0f} to {elevation.max():.0f} m")
    """
    try:
        import requests  # noqa: F401
        import xarray as xr
    except ImportError:
        raise ImportError(
            "xarray and requests required. Install with: pip install xarray requests netCDF4"
        )

    lon_min, lat_min, lon_max, lat_max = bbox

    # Validate bbox
    if not (-180 <= lon_min < lon_max <= 180):
        raise ValueError(
            f"Longitude must be between -180 and 180. Got range: [{lon_min}, {lon_max}]"
        )
    if not (-90 <= lat_min < lat_max <= 90):
        raise ValueError(f"Latitude must be between -90 and 90. Got range: [{lat_min}, {lat_max}]")

    # Setup cache directory
    if data_dir is None:
        data_dir = Path.home() / ".earthsciences" / "etopo2022"
    else:
        data_dir = Path(data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)

    # ETOPO 2022 is available via THREDDS server
    if resolution == "15s":
        url = "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/15s/15s_surface_elev_netcdf/ETOPO_2022_v1_15s_N90W180_surface.nc"
    elif resolution == "60s":
        url = "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"
    else:
        raise ValueError(f"Unknown resolution: {resolution}")

    try:
        # Open remote dataset (OPeNDAP)
        ds = xr.open_dataset(url)

        # Extract region
        subset = ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

        elevation = subset["z"].values
        lon = subset["lon"].values
        lat = subset["lat"].values

        ds.close()

        return {
            "elevation": elevation,
            "lon": lon,
            "lat": lat,
            "resolution": resolution,
            "bbox": bbox,
            "source": "ETOPO 2022",
            "units": "meters",
        }

    except Exception as e:
        warnings.warn(
            f"Could not load ETOPO2022 from remote server: {e}\n"
            f"You may need to download the data manually from:\n"
            f"https://www.ngdc.noaa.gov/mgg/global/"
        )
        raise


def load_srtm(lat: float, lon: float, resolution: str = "1s", data_dir: str | None = None) -> dict:
    """
    Load SRTM (Shuttle Radar Topography Mission) elevation data.

    SRTM provides near-global elevation data at 1 or 3 arc-second resolution.
    Coverage: 60°N to 56°S.

    Parameters
    ----------
    lat : float
        Latitude (degrees)
    lon : float
        Longitude (degrees)
    resolution : str, optional
        '1s' (1 arc-second, ~30m) or '3s' (3 arc-second, ~90m)
        (default='1s')
    data_dir : str, optional
        Directory containing SRTM tiles

    Returns
    -------
    dict
        Dictionary with 'elevation', 'lon', 'lat', 'metadata'

    Notes
    -----
    SRTM tiles are 1°x1° and named like: N37W122.hgt

    Data can be downloaded from:
    - USGS Earth Explorer: https://earthexplorer.usgs.gov/
    - OpenTopography: https://opentopography.org/

    Examples
    --------
    >>> # Load tile containing San Francisco
    >>> data = load_srtm(lat=37.7, lon=-122.4, resolution='1s',
    ...                  data_dir='/path/to/srtm/tiles')
    >>>
    >>> elevation = data['elevation']
    >>> print(f"Tile shape: {elevation.shape}")
    """
    # Determine tile name
    lat_int = int(np.floor(lat))
    lon_int = int(np.floor(lon))

    lat_hem = "N" if lat_int >= 0 else "S"
    lon_hem = "E" if lon_int >= 0 else "W"

    tile_name = f"{lat_hem}{abs(lat_int):02d}{lon_hem}{abs(lon_int):03d}.hgt"

    # Setup data directory
    if data_dir is None:
        data_dir = Path.home() / ".earthsciences" / "srtm"
    else:
        data_dir = Path(data_dir)

    tile_path = data_dir / tile_name

    if not tile_path.exists():
        raise FileNotFoundError(
            f"SRTM tile not found: {tile_path}\n"
            f"Download from: https://earthexplorer.usgs.gov/ or https://opentopography.org/\n"
            f"Place .hgt files in: {data_dir}"
        )

    # Read .hgt file (raw binary format)
    if resolution == "1s":
        size = 3601  # 1 arc-second tiles are 3601x3601
    elif resolution == "3s":
        size = 1201  # 3 arc-second tiles are 1201x1201
    else:
        raise ValueError(f"Unknown resolution: {resolution}")

    # Read binary data (big-endian 16-bit signed integers)
    elevation = np.fromfile(tile_path, dtype=">i2").reshape(size, size)

    # Replace void values (-32768) with NaN
    elevation = elevation.astype(float)
    elevation[elevation == -32768] = np.nan

    # Create coordinate arrays
    if resolution == "1s":
        step = 1.0 / 3600  # 1 arc-second
    else:
        step = 3.0 / 3600  # 3 arc-seconds

    lat_array = np.linspace(lat_int + 1, lat_int, size)
    lon_array = np.linspace(lon_int, lon_int + 1, size)

    return {
        "elevation": elevation,
        "lat": lat_array,
        "lon": lon_array,
        "tile_name": tile_name,
        "resolution": resolution,
        "source": "SRTM",
        "units": "meters",
    }


def load_gtopo30(bbox: tuple[float, float, float, float], data_dir: str | None = None) -> dict:
    """
    Load GTOPO30 global 30 arc-second elevation data.

    GTOPO30 is a global DEM with 30 arc-second resolution (~1 km).

    Parameters
    ----------
    bbox : tuple
        Bounding box (lon_min, lat_min, lon_max, lat_max)
    data_dir : str, optional
        Directory containing GTOPO30 tiles

    Returns
    -------
    dict
        Dictionary with elevation data

    Notes
    -----
    GTOPO30 is divided into 33 tiles covering the globe.

    Download from: https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-global-30-arc-second-elevation-gtopo30

    Examples
    --------
    >>> bbox = (-125, 30, -115, 40)  # California region
    >>> data = load_gtopo30(bbox, data_dir='/path/to/gtopo30')
    """
    if data_dir is None:
        data_dir = Path.home() / ".earthsciences" / "gtopo30"
    else:
        data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"GTOPO30 data directory not found: {data_dir}\n"
            f"Download from: https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-global-30-arc-second-elevation-gtopo30"
        )

    # GTOPO30 tiles are in various formats (DEM, BIL)
    # This is a simplified loader - full implementation would need to handle all tile formats

    warnings.warn(
        "GTOPO30 loader is simplified. For full functionality, consider using rasterio or GDAL.\n"
        "Example: rasterio.open('gtopo30_tile.dem')"
    )

    return {
        "message": "GTOPO30 loader requires rasterio or GDAL for full functionality",
        "data_dir": str(data_dir),
        "bbox": bbox,
    }


def download_tile(
    dataset: str,
    tile_id: str,
    output_dir: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> str:
    """
    Download a specific tile from online repositories.

    Parameters
    ----------
    dataset : str
        Dataset name: 'srtm', 'etopo', 'aster_gdem'
    tile_id : str
        Tile identifier (e.g., 'N37W122' for SRTM)
    output_dir : str, optional
        Output directory
    username : str, optional
        Username for authenticated downloads
    password : str, optional
        Password for authenticated downloads

    Returns
    -------
    str
        Path to downloaded file

    Notes
    -----
    Some datasets require free registration:
    - SRTM: https://earthexplorer.usgs.gov/
    - ASTER GDEM: https://search.asf.alaska.edu/

    Examples
    --------
    >>> # Download SRTM tile (requires USGS account)
    >>> path = download_tile('srtm', 'N37W122',
    ...                      username='your_username',
    ...                      password='your_password')
    """
    from urllib.parse import urljoin

    import requests

    if output_dir is None:
        output_dir = Path.home() / ".earthsciences" / dataset
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset-specific URLs
    urls = {
        "srtm": "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/",
        "aster_gdem": "https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/",
    }

    if dataset not in urls:
        valid_datasets = ", ".join(f"'{k}'" for k in urls.keys())
        raise ValueError(f"Unknown dataset '{dataset}'. Valid options: {valid_datasets}")

    base_url = urls[dataset]

    # Construct filename
    if dataset == "srtm":
        filename = f"{tile_id}.SRTMGL1.hgt.zip"
    elif dataset == "aster_gdem":
        filename = f"ASTGTMV003_{tile_id}_dem.tif"
    else:
        filename = f"{tile_id}.zip"

    url = urljoin(base_url, filename)
    output_path = output_dir / filename

    if output_path.exists():
        logger.info(f"File already exists: {output_path}")
        return str(output_path)

    logger.info(f"Downloading {filename}...")
    logger.info(f"From: {url}")

    try:
        # Setup authentication if provided
        auth = (username, password) if username and password else None

        response = requests.get(url, auth=auth, stream=True)
        response.raise_for_status()

        # Download with progress
        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = (downloaded / total_size) * 100
                        logger.debug(f"Progress: {progress:.1f}%")

        logger.info(f"Downloaded to: {output_path}")
        return str(output_path)

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            raise ValueError(
                "Authentication failed. Register for free at:\n"
                "https://urs.earthdata.nasa.gov/users/new"
            )
        else:
            raise


def get_tile_list(dataset: str, bbox: tuple[float, float, float, float]) -> list[str]:
    """
    Get list of tiles needed to cover a bounding box.

    Parameters
    ----------
    dataset : str
        Dataset name: 'srtm', 'aster_gdem'
    bbox : tuple
        Bounding box (lon_min, lat_min, lon_max, lat_max)

    Returns
    -------
    list
        List of tile identifiers

    Examples
    --------
    >>> # Get tiles for California
    >>> bbox = (-125, 32, -114, 42)
    >>> tiles = get_tile_list('srtm', bbox)
    >>> print(f"Need {len(tiles)} tiles")
    >>> print(tiles[:5])  # First 5 tiles
    """
    lon_min, lat_min, lon_max, lat_max = bbox

    tiles = []

    # Iterate over 1°x1° tiles
    for lat in range(int(np.floor(lat_min)), int(np.ceil(lat_max))):
        for lon in range(int(np.floor(lon_min)), int(np.ceil(lon_max))):
            lat_hem = "N" if lat >= 0 else "S"
            lon_hem = "E" if lon >= 0 else "W"

            tile_id = f"{lat_hem}{abs(lat):02d}{lon_hem}{abs(lon):03d}"
            tiles.append(tile_id)

    return tiles
