from pathlib import Path
from warnings import warn

import xarray as xr

from . import geometries
from .utils import _set_attrs_on_all_vars, _check_filetype, _separate_metadata


def open_grid(gridfilepath='./grid.nc', geometry=None, ds=None, quiet=False):
    """
    Opens a BOUT++ grid file.

    Opens a BOUT++ grid file (as generated by Hypnotoad etc.). Then sets
    coordinates depending on what's in the file, and moves grid data to the
    attributes dict.

    Parameters
    ----------
    gridfilepath : str, optional
    geometry : str, optional
        The geometry type of the grid data. This will specify what type of
        coordinates to add to the dataset, e.g. 'toroidal' or 'cylindrical'.

        If not specified then will attempt to read it from the file attrs.
        If still not found then a warning will be thrown, which can be
        suppressed by passing `quiet`=True.

        To define a new type of geometry you need to use the
        `register_geometry` decorator.
    ds : xarray.Dataset, optional
        BOUT dataset to merge grid information with.
        Leave unspecified if you just want to open the grid file alone.

    Returns
    -------
    ds : xarray.Dataset

    """

    gridfilepath = Path(gridfilepath)
    grid = xr.open_dataset(gridfilepath, engine=_check_filetype(gridfilepath))

    # TODO find out what 'yup_xsplit' etc are in the doublenull storm file John gave me
    # For now drop any variables with extra dimensions
    acceptable_dims = ['t', 'x', 'y', 'z']
    unrecognised_dims = list(set(grid.dims) - set(acceptable_dims))
    if len(unrecognised_dims) > 0:
        # Weird string formatting is a workaround to deal with possible bug in
        # pytest warnings capture - doesn't match strings containing brackets
        warn("Will drop all variables containing the dimensions {} because "
             "they are not recognised".format(str(unrecognised_dims)[1:-1]),
             UserWarning)
        grid = grid.drop_dims(unrecognised_dims)

    # Merge into one dataset, with scalar vars in attrs
    grid, grid_metadata = _separate_metadata(grid)
    if ds is None:
        ds = grid
    else:
        # TODO should instead drop variables which appear twice from grid
        # i.e. assume dataset variables are correct
        ds = xr.merge((ds, grid))

    ds = _set_attrs_on_all_vars(ds, 'grid', grid_metadata)

    if geometry is None:
        if geometry in ds.attrs:
            geometry = ds.attrs.get('geometry')
        else:
            if not quiet:
                warn("No geometry type found, no coordinates will be added")

    if geometry is not None:
        # Update coordinates to match particular geometry of grid
        ds = geometries.apply_geometry(ds, geometry)

    return ds
