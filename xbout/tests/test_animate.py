import pytest

from xbout import open_boutdataset
from xbout.boutdataarray import BoutDataArrayAccessor
from .test_load import create_bout_ds_list

from animatplot.blocks import Imshow, Line


@pytest.fixture
def create_test_file(tmpdir_factory):

    # Create temp dir for output of animate1D/2D
    save_dir = tmpdir_factory.mktemp("test_data")

    # Generate some test data
    ds_list, file_list = create_bout_ds_list("BOUT.dmp", nxpe=3, nype=3,
                                             syn_data_type="linear")
    for ds, file_name in zip(ds_list, file_list):
        ds.to_netcdf(str(save_dir.join(str(file_name))))

    ds = open_boutdataset(save_dir.join("BOUT.dmp.*.nc"))  # Open test data

    return save_dir, ds


class TestAnimate:
    """
    Set of tests to check whether animate1D() and animate2D() are running properly
    and PillowWriter is saving each animation correctly
    """
    def test_animate2D(self, create_test_file):

        save_dir, ds = create_test_file

        animation = ds['n'].isel(x=1).bout.animate2D(save_as="%s/testyz" % save_dir)

        assert isinstance(animation, Imshow)

        assert animation.ax.get_xlabel() == 'y'
        assert animation.ax.get_ylabel() == 'z'

        animation = ds['n'].isel(y=2).bout.animate2D(save_as="%s/testxz" % save_dir)

        assert isinstance(animation, Imshow)
        assert animation.ax.get_xlabel() == 'x'
        assert animation.ax.get_ylabel() == 'z'

        animation = ds['n'].isel(z=3).bout.animate2D(save_as="%s/testxy" % save_dir)

        assert isinstance(animation, Imshow)
        assert animation.ax.get_xlabel() == 'x'
        assert animation.ax.get_ylabel() == 'y'

    def test_animate2D_with_coords_and_dask(self, create_test_file):

        save_dir, ds = create_test_file

        import numpy as np
        import xarray as xr
        array = np.random.randint(0, 100, size=(3, 4, 5))
        da = xr.DataArray(array, dims=('t','x','y'),
                          coords={'t': [10, 20,30], 'x': [1, 2,3,4],
                                  'y': [100, 200,300,400,500]})
        dask_da = da.chunk(chunks={'t':3})
        animation = dask_da.bout.animate2D(animate_over='t', x='x', y='y',
                                           save_as="%s/test" % save_dir)
        assert isinstance(animation, Imshow)

    def test_animate1D(self, create_test_file):

        save_dir, ds = create_test_file
        animation = ds['n'].isel(y=2, z=0).bout.animate1D(save_as="%s/test" % save_dir)

        assert isinstance(animation, Line)
