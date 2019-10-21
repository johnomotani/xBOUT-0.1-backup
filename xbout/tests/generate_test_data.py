from netCDF4 import Dataset
from pathlib import Path
import numpy

def generate_test_data(datapath, nt, nx, ny, nz, NXPE=1, NYPE=1):
    """
    Generate some test files, imitating a set of BOUT++ output files.
    Note: as for BOUT++ output, nx=(# grid points)+2*MXG, while ny=(# grid points) with no
    boundary cells.
    """

    MXG = 2
    MYG = 2

    if (nx - 2*MXG)%NXPE != 0:
        raise ValueError("NXPE does not divide the number of grid points in x")
    if ny%NYPE != 0:
        raise ValueError("NYPE does not divide the number of grid points in y")

    mxsub = (nx - 2*MXG)//NXPE
    mysub = ny//NYPE

    localnx = mxsub + 2*MXG
    localny = mysub + 2*MYG

    datapath = Path(datapath)

    globaldata = numpy.arange((nx - 2*MXG)*ny*nz).reshape(((nx - 2*MXG), ny, nz))

    for yproc in range(NYPE):
        for xproc in range(NXPE):
            i = yproc*NXPE + xproc

            dataset = Dataset(datapath.joinpath('BOUT.dmp.'+str(i)+'.nc'), 'w')

            dataset.createDimension('t', nt)
            dataset.createDimension('x', localnx)
            dataset.createDimension('y', localny)
            dataset.createDimension('z', nz)

            def add_int(name, value):
                dataset.createVariable(name, 'i8')
                dataset[name][...] = value

            def add_float(name, value):
                dataset.createVariable(name, 'f8')
                dataset[name][...] = value

            def add_2d_constant(name, value):
                dataset.createVariable(name, 'f8', dimensions=('x', 'y'))
                dataset[name][...] = value

            def add_3d_constant(name, value):
                dataset.createVariable(name, 'f8', dimensions=('x', 'y', 'z'))
                dataset[name][...] = value

            def add_2d(name, value):
                dataset.createVariable(name, 'f8', dimensions=('t', 'x', 'y'))
                dataset[name][...] = value

            def add_3d(name, value):
                dataset.createVariable(name, 'f8', dimensions=('t', 'x', 'y', 'z'))
                dataset[name][...] = value

            add_float('BOUT_VERSION', 4.30)

            add_int('NXPE', NXPE)
            add_int('NYPE', NYPE)
            add_int('NZPE', 1)
            add_int('PE_XIND', xproc)
            add_int('PE_YIND', yproc)
            add_int('MYPE', i)

            add_int('nx', nx)
            add_int('ny', ny)
            add_int('nz', nz)
            add_int('MZ', nz)
            add_int('MXG', MXG)
            add_int('MYG', MYG)
            add_int('MZG', 0)
            add_int('MXSUB', mxsub)
            add_int('MYSUB', mysub)
            add_int('MZSUB', nz)
            add_int('ixseps1', nx)
            add_int('ixseps2', nx)
            add_int('jyseps1_1', 0)
            add_int('jyseps2_2', ny)
            add_int('jyseps2_1', ny//2 - 1)
            add_int('jyseps1_2', ny//2 - 1)
            add_int('ny_inner', ny//2)

            one = numpy.ones([localnx, localny])
            zero = numpy.zeros([localnx, localny])

            add_int('zperiod', 1)
            add_float('ZMIN', 0.)
            add_float('ZMAX', 2.*numpy.pi)
            add_2d_constant('g11', one)
            add_2d_constant('g22', one)
            add_2d_constant('g33', one)
            add_2d_constant('g12', zero)
            add_2d_constant('g13', zero)
            add_2d_constant('g23', zero)
            add_2d_constant('g_11', one)
            add_2d_constant('g_22', one)
            add_2d_constant('g_33', one)
            add_2d_constant('g_12', zero)
            add_2d_constant('g_13', zero)
            add_2d_constant('g_23', zero)
            add_2d_constant('G1', zero)
            add_2d_constant('G2', zero)
            add_2d_constant('G3', zero)

            add_2d_constant('J', one)
            add_2d_constant('Bxy', one)
            add_2d_constant('zShift', zero)

            add_2d_constant('dx', one*.5)
            add_2d_constant('dy', one*2.)
            add_float('dz', .7)

            add_int('iteration', nt)
            dataset.createVariable('t_array', 'f8', dimensions=('t'))
            dataset['t_array'][...] = numpy.arange(nt, dtype=float)*10.

            f3d = numpy.zeros((localnx, localny, nz))
            # set guard cells to NaN
            f3d[:MXG, :, :] = numpy.nan
            f3d[localnx-MXG:, :, :] = numpy.nan
            f3d[:, :MYG, :] = numpy.nan
            f3d[:, localny-MYG:, :] = numpy.nan

            f3d[MXG:localnx-MXG, MYG:localny-MYG, :] = globaldata[
                    xproc*mxsub:(xproc+1)*mxsub, yproc*mysub:(yproc+1)*mysub, :]

            if xproc == 0:
                # has inner x-boundary
                f3d[:MXG, MYG:localny-MYG, :] = -numpy.arange(
                    MXG*mysub*nz).reshape((MXG, mysub, nz))

            if xproc == NXPE - 1:
                # has outer x-boundary
                f3d[localnx-MXG:, MYG:localny-MYG, :] = -numpy.arange(
                    MXG*mysub*nz).reshape((MXG, mysub, nz))

            # no y-boundaries since ixseps1=nx, so only core region included

            add_3d_constant('f3d', f3d)

            f3d_evol = (f3d[numpy.newaxis, :, :, :]
                + numpy.zeros(nt)[:, numpy.newaxis, numpy.newaxis, numpy.newaxis])
            f3d_evol[:, MXG:localnx-MXG, MYG:localny-MYG, :] += (
                (nx - 2*MXG)*ny*nz
                *numpy.arange(nt)[:, numpy.newaxis, numpy.newaxis, numpy.newaxis])
            add_3d('f3d_evol', f3d_evol)

            dataset.close()
