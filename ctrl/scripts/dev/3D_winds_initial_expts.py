import string
import sys
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, zoom

CHIL_X = 439285
CHIL_Y = 138620

basedir = Path('/gws/pw/j07/woest/rjthomps/winds3d/data/20230803')


def regrid_3D_winds_to_eastings_northings(ds3d, eastings=None, northings=None, interp_method='linear'):
    # Create natural eastings/northings coords that correspond to the domain near Chilbolton
    if eastings is None:
        eastings = np.arange(280500.0, 477501, 1000)
    if northings is None:
        northings = np.arange(31500.0, 195501.0, 1000)
    nz = len(ds3d.altitude)
    dsout = xr.Dataset(
        data_vars=dict(
            u=(('time', 'altitude', 'northings', 'eastings'), np.zeros((1, nz, len(northings), len(eastings)))),
            v=(('time', 'altitude', 'northings', 'eastings'), np.zeros((1, nz, len(northings), len(eastings)))),
            w=(('time', 'altitude', 'northings', 'eastings'), np.zeros((1, nz, len(northings), len(eastings)))),
        ),
        coords=dict(
            time=('time', [0]),
            altitude=ds3d.altitude,
            northings=('northings', northings, {'units': 'm'}),
            eastings=('eastings', eastings, {'units': 'm'}),
        ),
    )

    # Construct data needed for interp.
    ee, nn = np.meshgrid(eastings, northings)
    osgb_tx = ccrs.OSGB()
    pc_tx = ccrs.PlateCarree()
    # Transform all lat/lon points into eastings, northings
    osgb_x, osgb_y, _ = osgb_tx.transform_points(
        pc_tx, ds3d.longitude.values.flatten(), ds3d.latitude.values.flatten()
    ).T
    points = np.array(list(zip(osgb_x, osgb_y)))

    # Do interp for each var for each height.
    for var in ['u', 'v', 'w']:
        print(var)
        for alt_idx in range(len(ds3d.altitude)):
            dsout[var][0, alt_idx] = griddata(
                points, ds3d[var].isel(altitude=alt_idx).values.flatten(), (ee, nn), method=interp_method
            )
    return dsout


def plot_datasets(ds_rad, ds3d_osgb, threed_winds_path, camra_path):
    az = ds_rad.rhi_mean_az.values.item()
    # Transect corresponding to RHI.
    r = np.arange(151) * 1e3
    transect_x = xr.DataArray(CHIL_X + r * np.sin(az * np.pi / 180), dims='transect')
    transect_y = xr.DataArray(CHIL_Y + r * np.cos(az * np.pi / 180), dims='transect')
    # Get u, v, w in the plane of the RHI.
    w_plane = ds3d_osgb.w.interp(eastings=transect_x, northings=transect_y)
    u_plane = ds3d_osgb.u.interp(eastings=transect_x, northings=transect_y)
    v_plane = ds3d_osgb.v.interp(eastings=transect_x, northings=transect_y)
    # Calc wind parallel to the plane/beam.
    hor_wind_plane = u_plane * np.sin(az * np.pi / 180) + v_plane * np.cos(az * np.pi / 180)
    hor_div = (hor_wind_plane[:, :, 2:] - hor_wind_plane[:, :, :-2]) / 2e3

    # Set up a grid to interp the 2D fields onto to give some context for the 2D fields.
    rgrid = np.linspace(-10, 150, 161) * 1e3
    rstart = np.linspace(-10, 10, 21) * 1e3
    xstart = CHIL_X - rstart * np.cos(az * np.pi / 180)
    ystart = CHIL_Y + rstart * np.sin(az * np.pi / 180)

    xcoords = []
    ycoords = []
    for xs, ys in zip(xstart, ystart):
        x = xs + rgrid * np.sin(az * np.pi / 180)
        y = ys + rgrid * np.cos(az * np.pi / 180)
        xcoords.append(x)
        ycoords.append(y)

    xcoords = xr.DataArray(np.array(xcoords), dims=['xrot', 'yrot'])
    ycoords = xr.DataArray(np.array(ycoords), dims=['xrot', 'yrot'])

    fig, axes = plt.subplots(
        2, 2, sharex=True, sharey='row', figsize=(15, 12), layout='constrained', height_ratios=[2, 2]
    )
    ((ax0, ax1), (ax2, ax3)) = axes

    # Plot w at ~2km near beam (interped onto xcoords/ycoords grid).
    im = ax0.pcolormesh(
        rgrid / 1e3,
        rstart / 1e3,
        ds3d_osgb.w.sel(altitude=2000, method='nearest').interp(eastings=xcoords, northings=ycoords).isel(time=0),
        vmin=-3,
        vmax=3,
        cmap='bwr',
    )
    plt.colorbar(im, ax=ax0, orientation='horizontal', pad=0.03, aspect=75, label='$w$ [m s$^{-1}$]')
    ax4 = ax0.twinx()
    # Plot line graph of w along beam.
    ax4.plot(
        rgrid[10:] / 1e3,
        ds3d_osgb.w.sel(altitude=2000, method='nearest')
        .interp(eastings=xcoords[10, 10:], northings=ycoords[10, 10:])
        .isel(time=0),
    )

    # Plot RadarNet/Nimrod near beam.
    im = ax1.pcolormesh(
        rgrid / 1e3, rstart / 1e3, ds_rad.nimrod_flow_interped_rain.interp(eastings=xcoords, northings=ycoords)
    )
    plt.colorbar(im, ax=ax1, orientation='horizontal', pad=0.03, aspect=75, label='rain [mm h$^{-1}$]')

    ax5 = ax1.twinx()
    # Plot line graph of RadarNet along beam.
    ax5.plot(
        rgrid[10:] / 1e3, ds_rad.nimrod_flow_interped_rain.interp(eastings=xcoords[10, 10:], northings=ycoords[10, 10:])
    )

    # Visual representation of beam.
    rplot = np.linspace(0, 150, 151)
    for ax in [ax0, ax1]:
        ax.plot(rplot, np.zeros(151), 'k--')
        ax.plot(rplot[0], 0, 'ko')
        ax.scatter(rplot[20::5], np.zeros(151)[20::5], c='k', marker='+', s=20)
        ax.set_ylim(-10, 10)

    az = ds_rad.rhi_mean_az.values.item()

    # Little north arrow (could be better).
    xend = 2.5 * np.sin(-(az - 90) * np.pi / 180)
    yend = 2.5 * np.cos(-(az - 90) * np.pi / 180)
    for ax in [ax0, ax1]:
        axins = ax.inset_axes([0.7, 0.7, 0.3, 0.3])
        axins.set_aspect('equal')
        axins.arrow(-xend, -yend, xend, yend, width=0.2, facecolor='k')
        axins.set_xlim(-3, 3)
        axins.set_ylim(-3, 3)
        axins.axis('off')
        axins.patch.set_alpha(0.5)

    ax4.set_ylim(-3, 3)
    ax5.set_ylim(0, None)

    ax0.set_ylabel('cross beam [km]')
    ax4.set_ylabel('$w$ [m s$^{-1}$]')
    ax5.set_ylabel('rain [mm h$^{-1}$]')

    # Plot w_plane, divergence in plane, u/w quivers, and Z 10 dBz.
    im = ax2.pcolormesh(r / 1e3, ds3d_osgb.altitude / 1e3, w_plane.isel(time=0), vmin=-3, vmax=3, cmap='bwr')
    plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.03, aspect=75, label='$w$ [m s$^{-1}$]')

    # Smooth noisy div.
    cs = ax2.contour(
        zoom(r[1:-1] / 1e3, 2),
        zoom(ds3d_osgb.altitude / 1e3, 2),
        zoom(-hor_div.isel(time=0).values, 2) * 1e4,
        levels=[-4, -1, 1, 4],
        linestyles=None,
        negative_linestyles='dashed',
        colors='k',
    )
    ax2.clabel(cs, cs.levels, inline=True, fontsize=9)
    ax2.set_xlim(60, 85)
    ax2.set_ylim(0, 8.5)
    q = ax2.quiver(
        r / 1e3, ds3d_osgb.altitude / 1e3, hor_wind_plane.isel(time=0), w_plane.isel(time=0) * 5, scale=200, pivot='mid'
    )
    ax2.quiverkey(q, X=0.7, Y=1.02, U=4, label='hor: 5 m s$^{-1}$, vert: 1 m s$^{-1}$', labelpos='E')
    # Smooth noisy Z for contours.
    # Important to set nans to zero otherwise smoothed contour will have breaks in it.
    Z_contour_vals = ds_rad.rhi_Z.values.copy()
    Z_contour_vals[np.isnan(Z_contour_vals)] = 0
    ax2.contour(ds_rad.x, ds_rad.z, gaussian_filter(Z_contour_vals, 2), levels=[10], linewidths=3)

    # Plot radar, and w contours.
    im1 = ax3.pcolormesh(ds_rad.x, ds_rad.z, ds_rad.rhi_Z, vmin=-10, vmax=60)
    plt.colorbar(im1, ax=ax3, orientation='horizontal', pad=0.03, aspect=75, label='reflectivity [dBZ]')

    ax3.contour(ds_rad.x, ds_rad.z, gaussian_filter(Z_contour_vals, 2), levels=[10], linewidths=3)
    ax3.contour(
        zoom(r / 1e3, 2),
        zoom(ds3d_osgb.altitude / 1e3, 2),
        zoom(w_plane.isel(time=0), 2),
        levels=[-1, -0.5, 0.5, 1],
        cmap='bwr',
    )

    ax2.grid(ls='--', lw=0.5, c='k')
    ax3.grid(ls='--', lw=0.5, c='k')

    ax2.set_ylabel('Height [km]')
    ax2.set_xlabel('Range [km]')
    ax3.set_xlabel('Range [km]')

    fig.align_ylabels(axes[:, 0])

    fig.suptitle('WesCon IOP 3/8/2023')

    w_alt = ds3d_osgb.sel(altitude=2000, method='nearest').altitude.values.item()
    camra_time = pd.Timestamp(ds_rad.time.values.item())
    tstr = camra_time.strftime('%H:%M:%S')
    titles = [
        f'Along-beam $w$ at {w_alt:.0f} m',
        'Along-beam RadarNet rain',
        '3D winds along transect at 13:10:00',
        f'CAMRa RHI at {tstr}',
    ]

    for i, ax in enumerate(axes.flatten()):
        title = titles[i]
        c = string.ascii_lowercase[i]
        ax.set_title(f'{c}) {title}', loc='left')

    outfilepath = Path(
        f'/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_figs/3D_winds/wescon.{threed_winds_path.stem}.CAMRa_{tstr}.png'
    )
    outfilepath.parent.mkdir(parents=True, exist_ok=True)
    print(outfilepath)
    plt.savefig(outfilepath)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        threed_winds_path = Path(sys.argv[1])
        camra_path = Path(sys.argv[2])
    else:
        # Path('/gws/pw/j07/woest/rjthomps/winds3d/data/20230803/grid_1000m_filter_1_0_20230803_1310_v3_Cw4.nc'))
        threed_winds_path = Path(
            '/gws/pw/j07/woest/rjthomps/winds3d/data/20230803/grid_1000m_filter_1_0_20230803_1310.nc'
        )
        camra_path = Path(
            '/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_output/wescon_radar_dev/20230803/camra/gridded_ncas-radar-camra-1_cao_20230803-131042_srhi_l1_v1.0.0.nc'
        )

    try:
        # Allows interactive running in ipython with run -i and no reload of data.
        if threed_winds_path != old_threed_winds_path:
            raise NameError()
        elif camra_path != old_camra_path:
            raise NameError()
        print('Use in-memory objs')
    except NameError:
        print('Load objs')
        # Close in time to the RHI scan I want to compare with.
        print('* open 3D winds')
        ds3d = xr.open_dataset(threed_winds_path)
        print('* open gridded data')
        ds_rad = xr.open_dataset(camra_path).sel(time='2023-08-03 13:10:42', method='nearest')

        print('* regrid 3D winds')
        ds3d_osgb = regrid_3D_winds_to_eastings_northings(ds3d)
        print('Loaded objs')
        old_threed_winds_path = threed_winds_path
        old_camra_path = camra_path

    print('plot data')
    plot_datasets(ds_rad, ds3d_osgb, threed_winds_path, camra_path)
