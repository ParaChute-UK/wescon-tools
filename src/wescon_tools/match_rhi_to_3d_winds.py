import string
from pathlib import Path

import cartopy.crs as ccrs
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, zoom

CHIL_X = 439285
CHIL_Y = 138620


class MatchRHIto3dWinds:
    def __init__(self, ds_rad, time_interp=True, winddir=Path('/gws/pw/j07/woest/rjthomps/winds3d/data/'),
                 filetpl='%Y%m%d/grid_1000m_filter_1_0_%Y%m%d_%H%M_v6.1.nc'):
        self.ds_rad = ds_rad
        self.time_interp = time_interp
        self.winddir = winddir
        self.filetpl = filetpl
        self.az = ds_rad.rhi_mean_az.values.item()
        self.r = np.arange(151) * 1e3

        self.paths_3d_winds = None

        self.w_plane = None
        self.u_plane = None
        self.v_plane = None
        self.w_plane_hr = None
        self.u_plane_hr = None
        self.v_plane_hr = None

        self.hor_div = None
        self.hor_wind_plane = None

    def match(self):
        rhi_time = pd.Timestamp(self.ds_rad.time.values.item())
        if self.time_interp:
            logger.info('* open 3D winds time interp')
            paths = self.find_closest_3d_winds(rhi_time)
            self.paths_3d_winds = paths
            ds_winds = xr.open_mfdataset(paths, combine='nested', concat_dim='time')
            self.ds3d = ds_winds.interp(time=rhi_time).compute()
        else:
            logger.info('* open 3D winds')
            path = self.find_closest_3d_winds(rhi_time, method='nearest')
            self.paths_3d_winds = [path]
            self.ds3d = xr.open_dataset(path).isel(time=0).compute()

        logger.info('* regrid 3D winds')
        self.ds3d_osgb = self.regrid_3d_winds_to_eastings_northings(self.ds3d)
        self.calc_3d_winds_transects()

    def find_closest_3d_winds(self, rhi_time, method='either_side'):
        if method == 'either_side':
            wind_time1 = rhi_time.floor('10min')
            wind_time2 = rhi_time.ceil('10min')
            # '20230803/grid_1000m_filter_1_0_20230803_1310_v6.1.nc'
            file_path1 = self.winddir / wind_time1.strftime(self.filetpl)
            file_path2 = self.winddir / wind_time2.strftime(self.filetpl)
            return file_path1, file_path2
        elif method == 'nearest':
            wind_time1 = rhi_time.round('10min')
            file_path1 = self.winddir / wind_time1.strftime(self.filetpl)
            return file_path1
        else:
            raise Exception('method must be either_side or nearest')

    def regrid_3d_winds_to_eastings_northings(self, ds3d, eastings=None, northings=None, interp_method='linear'):
        # Create natural eastings/northings coords that correspond to the domain near Chilbolton
        if eastings is None:
            eastings = np.arange(280500.0, 477501, 1000)
        if northings is None:
            northings = np.arange(31500.0, 195501.0, 1000)
        nz = len(ds3d.altitude)
        dsout = xr.Dataset(
            data_vars=dict(u=(('altitude', 'northings', 'eastings'), np.zeros((nz, len(northings), len(eastings)))),
                           v=(('altitude', 'northings', 'eastings'), np.zeros((nz, len(northings), len(eastings)))),
                           w=(('altitude', 'northings', 'eastings'), np.zeros((nz, len(northings), len(eastings)))), ),
            coords=dict(altitude=ds3d.altitude, northings=('northings', northings, {'units': 'm'}),
                        eastings=('eastings', eastings, {'units': 'm'}), ), )

        # Construct data needed for interp.
        ee, nn = np.meshgrid(eastings, northings)
        osgb_tx = ccrs.OSGB()
        pc_tx = ccrs.PlateCarree()
        # Transform all lat/lon points into eastings, northings
        osgb_x, osgb_y, _ = osgb_tx.transform_points(pc_tx, ds3d.longitude.values.flatten(),
                                                     ds3d.latitude.values.flatten()).T
        points = np.array(list(zip(osgb_x, osgb_y)))

        # Do interp for each var for each height.
        for var in ['u', 'v', 'w']:
            logger.info(var)
            for alt_idx in range(len(ds3d.altitude)):
                dsout[var][alt_idx] = griddata(points, ds3d[var].isel(altitude=alt_idx).values.flatten(), (ee, nn),
                                               method=interp_method)
        return dsout

    def calc_3d_winds_transects(self):
        # Transect corresponding to RHI.
        transect_x = xr.DataArray(CHIL_X + self.r * np.sin(self.az * np.pi / 180), dims='transect',
                                  coords={'transect': self.r})
        transect_y = xr.DataArray(CHIL_Y + self.r * np.cos(self.az * np.pi / 180), dims='transect',
                                  coords={'transect': self.r})
        # transect_x = xr.DataArray(CHIL_X + self.r * np.sin(self.az * np.pi / 180), dims='transect')
        # transect_y = xr.DataArray(CHIL_Y + self.r * np.cos(self.az * np.pi / 180), dims='transect')
        # Define the coordinate here
        # Get u, v, w in the plane of the RHI.
        self.w_plane = self.ds3d_osgb.w.interp(eastings=transect_x, northings=transect_y)
        self.u_plane = self.ds3d_osgb.u.interp(eastings=transect_x, northings=transect_y)
        self.v_plane = self.ds3d_osgb.v.interp(eastings=transect_x, northings=transect_y)

        # Oversample w_plane to exactly the coords of the RHI (_hr == high-res).
        x = self.ds_rad.x.values * 1000
        z = self.ds_rad.z.values * 1000
        self.w_plane_hr = self.w_plane.interp(transect=x, altitude=z)
        self.u_plane_hr = self.u_plane.interp(transect=x, altitude=z)
        self.v_plane_hr = self.v_plane.interp(transect=x, altitude=z)
        # Calc wind parallel to the plane/beam.
        self.hor_wind_plane = self.u_plane * np.sin(self.az * np.pi / 180) + self.v_plane * np.cos(
            self.az * np.pi / 180)
        self.hor_div = (self.hor_wind_plane.values[:, 2:] - self.hor_wind_plane.values[:, :-2]) / 2e3


class Plot3dWinds:
    def __init__(self, matcher):
        self.matcher = matcher
        self.ds_rad = matcher.ds_rad
        self.ds3d_osgb = matcher.ds3d_osgb
        self.az = self.ds_rad.rhi_mean_az.values.item()
        self.r = matcher.r

        self.xcoords = None
        self.ycoords = None

        self.gen_2d_grid()

    def gen_2d_grid(self):
        # Set up a grid to interp the 2D fields onto to give some context for the 2D fields.
        self.rgrid = np.linspace(-10, 150, 161) * 1e3
        self.rstart = np.linspace(-10, 10, 21) * 1e3
        xstart = CHIL_X - self.rstart * np.cos(self.az * np.pi / 180)
        ystart = CHIL_Y + self.rstart * np.sin(self.az * np.pi / 180)

        xcoords = []
        ycoords = []
        for xs, ys in zip(xstart, ystart):
            x = xs + self.rgrid * np.sin(self.az * np.pi / 180)
            y = ys + self.rgrid * np.cos(self.az * np.pi / 180)
            xcoords.append(x)
            ycoords.append(y)
        self.xcoords = xr.DataArray(np.array(xcoords), dims=['xrot', 'yrot'])
        self.ycoords = xr.DataArray(np.array(ycoords), dims=['xrot', 'yrot'])

    def plot(self):
        fig, axes = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(15, 12), layout='constrained',
                                 height_ratios=[2, 2])
        ((ax0, ax1), (ax2, ax3)) = axes

        ax4 = self._plot_2d_w_context(ax0)
        ax5 = self._plot_2d_radarnet_context(ax1)

        self._add_beam(ax0, ax1)
        self._add_north_arrow(ax0, ax1)

        Z_contour_vals = self._plot_3d_winds_vert_cross_section(ax2)
        self._plot_rhi_with_contours(ax3, Z_contour_vals)

        self._limits_and_labels(ax0, ax2, ax3, ax4, ax5)
        self._titles_and_layout(fig, axes)

    def _plot_2d_w_context(self, ax0):
        # Plot w at ~2km near beam (interped onto xcoords/ycoords grid).
        im = ax0.pcolormesh(self.rgrid / 1e3, self.rstart / 1e3,
                            self.ds3d_osgb.w.sel(altitude=2000, method='nearest').interp(eastings=self.xcoords,
                                                                                         northings=self.ycoords),
                            vmin=-3, vmax=3, cmap='bwr', )
        plt.colorbar(im, ax=ax0, orientation='horizontal', pad=0.03, aspect=75, label='$w$ [m s$^{-1}$]')
        ax4 = ax0.twinx()
        # Plot line graph of w along beam.
        ax4.plot(self.rgrid[10:] / 1e3,
                 self.ds3d_osgb.w.sel(altitude=2000, method='nearest').interp(eastings=self.xcoords[10, 10:],
                                                                              northings=self.ycoords[10, 10:]))
        return ax4

    def _plot_2d_radarnet_context(self, ax1):
        # Plot RadarNet/Nimrod near beam.
        im = ax1.pcolormesh(self.rgrid / 1e3, self.rstart / 1e3,
                            self.ds_rad.radarnet_flow_interped_rain.interp(eastings=self.xcoords,
                                                                           northings=self.ycoords))
        plt.colorbar(im, ax=ax1, orientation='horizontal', pad=0.03, aspect=75, label='rain [mm h$^{-1}$]')

        ax5 = ax1.twinx()
        # Plot line graph of RadarNet along beam.
        ax5.plot(self.rgrid[10:] / 1e3, self.ds_rad.radarnet_flow_interped_rain.interp(eastings=self.xcoords[10, 10:],
                                                                                       northings=self.ycoords[10, 10:]))
        return ax5

    def _add_beam(self, ax0, ax1):
        # Visual representation of beam.
        rplot = np.linspace(0, 150, 151)
        for ax in [ax0, ax1]:
            ax.plot(rplot, np.zeros(151), 'k--')
            ax.plot(rplot[0], 0, 'ko')
            ax.scatter(rplot[20::5], np.zeros(151)[20::5], c='k', marker='+', s=20)
            ax.set_ylim(-10, 10)

    def _add_north_arrow(self, ax0, ax1):
        az = self.az

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

    def _plot_3d_winds_vert_cross_section(self, ax2):
        # Plot w_plane, divergence in plane, u/w quivers, and Z 10 dBz.
        # im = ax2.pcolormesh(self.w_plane_hr.transect, self.w_plane_hr.altitude, self.w_plane_hr, vmin=-3, vmax=3,
        # cmap='bwr')
        im = ax2.pcolormesh(self.matcher.w_plane.transect / 1e3, self.matcher.w_plane.altitude / 1e3,
                            self.matcher.w_plane, vmin=-3, vmax=3, cmap='bwr')
        plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.03, aspect=75, label='$w$ [m s$^{-1}$]')

        # Smooth noisy div.
        cs = ax2.contour(zoom(self.r[1:-1] / 1e3, 2), zoom(self.ds3d_osgb.altitude / 1e3, 2),
                         zoom(-self.matcher.hor_div, 2) * 1e4,  # self.r[1:-1],
                         # ds3d_osgb.altitude / 1e3,
                         # -self.hor_div.isel(time=0).values * 1e4,
                         levels=[-4, -1, 1, 4], linestyles=None, negative_linestyles='dashed', colors='k', )
        ax2.clabel(cs, cs.levels, inline=True, fontsize=9)
        ax2.set_xlim(60, 85)
        ax2.set_ylim(0, 8.5)
        q = ax2.quiver(self.r / 1e3, self.ds3d_osgb.altitude / 1e3, self.matcher.hor_wind_plane,
                       self.matcher.w_plane * 5, scale=200, pivot='mid')
        ax2.quiverkey(q, X=0.7, Y=1.02, U=4, label='hor: 5 m s$^{-1}$, vert: 1 m s$^{-1}$', labelpos='E')
        # Smooth noisy Z for contours.
        # Important to set nans to zero otherwise smoothed contour will have breaks in it.
        Z_contour_vals = self.ds_rad.rhi_Z.values.copy()
        Z_contour_vals[np.isnan(Z_contour_vals)] = 0
        ax2.contour(self.ds_rad.x, self.ds_rad.z, gaussian_filter(Z_contour_vals, 2), levels=[10], linewidths=3)
        return Z_contour_vals

    def _plot_rhi_with_contours(self, ax3, Z_contour_vals):
        # Plot radar, and w contours.
        im1 = ax3.pcolormesh(self.ds_rad.x, self.ds_rad.z, self.ds_rad.rhi_Z, vmin=-10, vmax=60)
        plt.colorbar(im1, ax=ax3, orientation='horizontal', pad=0.03, aspect=75, label='reflectivity [dBZ]')

        ax3.contour(self.ds_rad.x, self.ds_rad.z, gaussian_filter(Z_contour_vals, 2), levels=[10], linewidths=3)
        ax3.contour(zoom(self.r / 1e3, 2), zoom(self.ds3d_osgb.altitude / 1e3, 2), zoom(self.matcher.w_plane, 2),
                    levels=[-1, -0.5, 0.5, 1], cmap='bwr', )

    def _limits_and_labels(self, ax0, ax2, ax3, ax4, ax5):
        ax2.grid(ls='--', lw=0.5, c='k')
        ax3.grid(ls='--', lw=0.5, c='k')

        ax4.set_ylim(-3, 3)
        ax5.set_ylim(0, None)

        ax0.set_ylabel('cross beam [km]')
        ax4.set_ylabel('$w$ [m s$^{-1}$]')
        ax5.set_ylabel('rain [mm h$^{-1}$]')

        ax2.set_ylabel('Height [km]')
        ax2.set_xlabel('Range [km]')
        ax3.set_xlabel('Range [km]')

    def _titles_and_layout(self, fig, axes):
        fig.align_ylabels(axes[:, 0])

        fig.suptitle('WesCon IOP 3/8/2023')

        w_alt = self.ds3d_osgb.sel(altitude=2000, method='nearest').altitude.values.item()
        camra_time = pd.Timestamp(self.ds_rad.time.values.item())
        tstr = camra_time.strftime('%H:%M:%S')
        titles = [f'Along-beam $w$ at {w_alt:.0f} m', 'Along-beam RadarNet rain', '3D winds along transect at 13:10:00',
                  f'CAMRa RHI at {tstr}', ]

        for i, ax in enumerate(axes.flatten()):
            title = titles[i]
            c = string.ascii_lowercase[i]
            ax.set_title(f'{c}) {title}', loc='left')
