from pathlib import Path
from warnings import warn

import xarray as xr

from . import geometries
from .utils import _set_attrs_on_all_vars, _check_filetype, _separate_metadata


def open_grid(gridfilepath='./grid.nc', geometry=None, coordinates=None, ds=None,
              quiet=False, keep_xboundaries=False, keep_yboundaries=False):
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
    coordinates : sequence of str, optional
        Names to give the physical coordinates corresponding to 'x', 'y' and 'z' (in
        order). If not specified, default names are chosen.
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
    unrecognised_dims = set(grid.dims) - set(acceptable_dims)
    if len(unrecognised_dims) > 0:
        warn("Will drop all variables containing the dimensions {} because"
             "they are not recognised".format(unrecognised_dims))
        grid = grid.drop_dims(unrecognised_dims)

    if not keep_xboundaries:
        if ds is not None:
            xboundaries = int(ds.metadata['MXG'])
        else:
            xboundaries = 0
        if xboundaries > 0:
            grid = grid.isel(x=slice(xboundaries, -xboundaries, None))
    if not keep_yboundaries:
        yboundaries = int(grid['y_boundary_guards'])
        if yboundaries > 0:
            # Remove y-boundary cells from first divertor target
            grid = grid.isel(y=slice(yboundaries, -yboundaries, None))
            if grid['jyseps1_2'] > grid['jyseps2_1']:
                # There is a second divertor target, remove y-boundary cells there too
                nin = int(grid['ny_inner'])
                grid_lower = grid.isel(y=slice(None, nin, None))
                grid_upper = grid.isel(y=slice(nin+2*yboundaries, None, None))
                grid = xr.concat((grid_lower, grid_upper), dim='y', data_vars='minimal',
                                 compat='identical')

    # Merge into one dataset, with scalar vars in attrs
    grid, grid_metadata = _separate_metadata(grid)
    if ds is None:
        ds = grid
    else:
        if grid.get('y_boundary_guards', 0) > 0 and not ds.metadata['keep_yboundaries']:
            raise NotImplementedError('Do not know what to do with y-boundary cells in '
                                      'grid when dataset does not have y boundary cells')
        # Drop variables in ds from grid, so that variables saved to both do not conflict
        # when merging.
        # Prefer the version from ds, as this may have had normalisations, etc., applied
        # by the PhysicsModel that we want to keep.
        grid = grid.drop(ds.keys(), errors='ignore')
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
        ds = geometries.apply_geometry(ds, geometry, coordinates=coordinates)

    return ds
