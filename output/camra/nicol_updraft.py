import numpy as np
import matplotlib.pyplot as plt
from input.camra.load_camra_l1 import CamraLoader as CLP


class CamraUpdrafts:

    def __init__(self, camra_obj):
        self.camra_scan = camra_obj

        self.rVr = None

    def calc_rVr(self, Zh, Vr, r, Z_thresh):
        """
        Calculate the range scaled radial velocity rVr

        :param Zh: reflectivity
        :param Vr: radial velocity
        :param r: ranges
        :param Z_thresh: reflectivity threshold
        :return: range scaled radial velocity
        """

        indnan = np.where(Zh < Z_thresh)  # indices below threshold
        Vr[indnan] = np.nan  # replace values less than Z_thresh with nan
        rVr = Vr * (r / 1000)

        self.rVr = rVr

    def interp_data(
        self,
        x,
        y,
        data,
        xmin,
        xmax,
        ymin,
        ymax,
        dx=0.5,
        dy=0.5,
        method="linear",
        fillval=np.nan,
    ):
        """
        Function to interpolate data onto regular Cartesian grid
        Used for PPI and RHI scans

        :param x: x co-ords
        :param y: y co-ords
        :param data: data
        :param xmin: min x co-ord
        :param xmax: max x co-ord
        :param ymin: min y co-ord
        :param ymax: max y-cord
        :param dx: grid spacing in x or r
        :param dy: grid spacing in y or z
        :param method: method of interpolate: linear, cubic
        :param fillval: fill value for data outside of points
        :return: x, y co-ords and data on regular grid
        """

        from scipy.interpolate import griddata

        # define cartesian grid
        xr = np.arange(xmin, xmax, dx)
        yr = np.arange(ymin, ymax, dy)

        # make meshgrid
        xc, yc = np.meshgrid(xr, yr)

        # have to flatten the data to use griddata
        xcnew = xc.flatten()
        ycnew = yc.flatten()

        # do the interpolation
        xnew = x.flatten()
        ynew = y.flatten()
        datanew = data.flatten()
        out = griddata(
            (xnew, ynew), datanew, (xcnew, ycnew), method=method, fill_value=fillval
        )
        # reshape data back to shape of co-ords we need
        newout = out.reshape(xc.shape)

        return xc, yc, newout

    def reg_vert_grid(self, slc, zmin, zmax, fdz, dz):
        """
        Grids data from fine grid slice column to final regular grid.
        Based on method from John Nicol DYMECS project.
        NOTE: Currently does not work for certain resolutions with reshaping.

        :param data: data to grid to vertical regular grid, 2D array
        :param zmin: minimum z height
        :param zmax: max z height
        :param fdz: fine grid resolution
        :param dz: final grid resolution (usually coarser than fdz_
        :return: data re-gridded to dz resolution in vertical but still fine in horizontal
        """

        numh = int(zmax / dz)  # find num of vertical leves of regular grid
        hf = np.arange(zmin, zmax + fdz, fdz)  # define the fine grid
        hreg = np.arange(zmin, dz * (numh + 1), dz)  # define the regular grid

        slicereg = np.zeros([numh, np.shape(slc)[1]])  # init array to hold reg grid
        depth = np.zeros(np.shape(slicereg))  # init array to hold depths of layers

        # set the bases and tops in the fine grid and regular grid
        basereg = hreg[:-1]
        topreg = hreg[1:]
        base = hf[:-1]
        top = hf[1:]

        # find the lower and upper bounds of the box
        l = np.array([max([bf, bh]) for bf in basereg for bh in base])
        u = np.array([min([tf, th]) for tf in topreg for th in top])
        # find the differeneces between the bounds which is the depth of the box
        d = (u - l).reshape([numh, np.shape(slc)[0]])
        didx = np.where(d > 0)  # find indices of where d>0 i.e there is a box to grid

        for idx, fidx in enumerate(
            didx[0]
        ):  # loop through idxs, only need 0 as the array corresponds to correct shape
            if (
                fidx > 0
            ):  # this introduces the 0 slice at the bottom. Bottom and top of data are 0
                slicereg[fidx, 0:] = (
                    slicereg[fidx, 0:] + d[fidx, idx] * slc[idx, 0:]
                )  # calc the values for reg grid using d weighting
                depth[fidx, 0:] = (
                    depth[fidx, 0:] + d[fidx, idx]
                )  # add to depths array the depths for this slice

        slicereg = slicereg / depth  # calc final weighted data
        return slicereg

    def grid_rhi(
        self,
        data,
        r,
        z,
        dx,
        dz,
        fdx,
        fdz,
        xmin=0.0,
        xmax=180.0,
        zmin=0.0,
        zmax=12.0,
        method="linear",
        fill_value=np.nan,
    ):
        """
        Grid RHI data to a regular grid using SSAA in the horizontal and a regular gridding based on distance from slice
        tops and bottoms in the vertical.

        :param data: data to grid to regular rhi grid, 2d array
        :param r: ranges of data, 2d array
        :param z: z heights of data, 2d array
        :param dx: final horizontal grid res, km
        :param dz: final vertical grid res, km
        :param fdx: fine horiz. grid res, km
        :param fdz: fine vert. grid res., km
        :param xmin: min x bound of data, km, default=0km
        :param xmax: max x bound of data, km, default=180km
        :param zmin: min z bound of data, km, default=0km
        :param zmax: max z bound of data, km, default=12km
        :param method: method of interpolation for fine grid, default=linear
        :param fill_value: fill value for interpolation when points outside of bounds, default=nan
        :return: x, z and data on final regular grid
        """

        # Generate coarse grid
        xc = np.arange(xmin, xmax, dx)
        zc = np.arange(zmin, zmax, dz)
        xx, zz = np.meshgrid(xc, zc)
        # init arrays for final grid of data
        final_data = np.zeros(np.shape(xx)) + np.nan

        # interpolate data
        fxx, fzz, new_data = self.interp_data(
            r,
            z,
            data,
            xmin,
            xmax,
            zmin,
            zmax,
            dx=fdx,
            dy=fdz,
            method=method,
            fillval=fill_value,
        )

        # call function to regularly grid the data in the vertical but not horizontal
        new_data_vertreg = self.reg_vert_grid(new_data, zmin, zmax, fdz, dz)

        # now sample from high horizontal resolution grid to the final coarser resolution grid.
        # by taking the mean of a fixed window of high resolution points that are within the lower resolution grid box
        for i in range(0, len(xc)):
            for k in range(0, len(zc)):
                final_data[k, i] = np.nanmean(
                    new_data_vertreg[k, int(i * (dx / fdx)) : int(i * (dx / fdx) + 1)]
                )

        # return x, z and data on final cartesian grid
        # swap the x/z axes to use with retrieval function
        xx = np.swapaxes(xx, 0, 1)
        zz = np.swapaxes(zz, 0, 1)
        final_data = np.swapaxes(final_data, 0, 1)
        # mask out nans
        final_data = np.ma.array(final_data, mask=np.isnan(final_data))

        # Generate meshes of the top and right edges for plotting
        xc_edges = np.arange(xmin, xmax + dx, dx)
        zc_edges = np.arange(zmin, zmax + dz, dz)
        xx_edges, zz_edges = np.meshgrid(xc_edges, zc_edges, indexing="ij")

        return xx, zz, final_data, xx_edges, zz_edges

    def calc_rho_profile(self, z_list, mean_dz):
        """
        Function to calc density profile
        From John Nicol via email.

        :param z_list: list of z heights, kilometers
        :param mean_dz: mean dz, metres
        :return: density profile and d log(rho)\dz
        """

        T0 = 288.15  # sfc temp, K
        g = 9.80665  # gravity, ms^-2
        P0 = 101325.0  # sfc pressure, Pa
        L = 6.5  # lapse rate, K km^-1
        R = 8.31432  # gas constant for ideal gas
        M = 28.9644  # molar mass of air

        # convert from geometric height to geopotential height
        h = z_list
        E = 6356.766  # radius of Earth, km

        h = (E * h) / (E + h)

        T = T0 - L * h
        P = P0 * (1.0 - (L * h / T0)) ** ((g * M) / (R * L))
        rho = (P * M) / (R * T * 1000.0)
        dln_rho_dz = -1.0 * (np.diff(np.log(rho)) / mean_dz)

        self.rho_profile = rho
        self.dln_rho_dz = dln_rho_dz
        # return rho, dln_rho_dz

    def ground_up_vx_rhi(self, ZH, VEL, x, z, dx, dz, rho, Zedge=-10):
        """
        Function to integrate ground-up using eqn in terms of rho and calculating Vx at each height

        :param ZH: Reflectivity
        :param VEL: range scaled Doppler velocity
        :param x: x co-ords in Cartesian
        :param z: z co-ords in Cartesian
        :param dx: change in horizontal
        :param dz: change in height
        :param rho: density profile
        :param Zedge: reflectivity threshold for cloud edges, dBZ
        :return: Vz=velocity in vertical, Vx=velocity in horizontal, dvx=horizontal divergence
        """

        Vx = np.zeros(np.shape(VEL))
        Vz = np.zeros(np.shape(VEL))
        dvx = np.zeros(np.shape(VEL))
        Vx[:, :] = np.nan
        dvx[:, :] = np.nan

        for i in range(1, np.shape(VEL)[1]):  # go through z co-ords (height)
            Vx[:, i] = (VEL[:, i] - Vz[:, i - 1] * z[:, i]) / x[
                :, i
            ]  # calc Vx using Doppler vel eqn in cart. co-ords
            Vx[:, i][ZH[:, i] < Zedge] = np.nan  # mask Vx below reflectivity threshold
            for j in range(1, np.shape(VEL)[0] - 1):  # go through x (range)
                dVx = np.mean(
                    [Vx[j + 1, i] - Vx[j, i], Vx[j, i] - Vx[j - 1, i]]
                )  # calc. divergence using mean of differences
                dvx[j, i] = dVx
                if np.isfinite(dVx):  # only calc Vz if we have a value for dVx
                    Vz[j, i] = (rho[i - 1] / rho[i]) * Vz[j, i - 1] - (
                        ((rho[i] + rho[i - 1]) / (2.0 * rho[i])) * (dz / dx) * dVx
                    )

        # double check fields for cloud edges with Zedge threshold are filtered out
        edges = np.where(ZH < Zedge)
        Vz[edges] = np.nan
        Vx[edges] = np.nan
        dvx[edges] = np.nan
        Vz[np.where(np.isnan(VEL))] = (
            np.nan
        )  # where VEL are nans fill Vz with nans to remove 0's
        # mask nans
        Vz = np.ma.array(Vz, mask=np.isnan(Vz))
        Vx = np.ma.array(Vx, mask=np.isnan(Vx))
        dvx = np.ma.array(dvx, mask=np.isnan(dvx))

        self.ground_up_Vz = Vz
        self.ground_up_Vx = Vx
        self.ground_up_dvx = dvx
        # return Vz, Vx, dvx

    def top_down_vx_rhi(self, ZH, VEL, x, z, dx, dz, rho, Zedge=-10):
        """
        Function to integrate top down using eqn. in terms of rho and calculating vx at each height.

        :param ZH: Reflectivity
        :param VEL: Range scaled Doppler velocity
        :param x: x co-ords in Cartesian
        :param z: z co-ords in Cartesian
        :param dx: Change in horizontal
        :param dz: Change in height
        :param rho: density profile
        :param Zedge: reflectivity threshold for cloud edges, dBZ
        :return: Vz=vertical velocity, Vx=horizontal velocity, dvx = horizontal divergence
        """

        Vx = np.zeros(np.shape(VEL))
        Vz = np.zeros(np.shape(VEL))
        dvx = np.zeros(np.shape(VEL))
        Vx[:, :] = np.nan
        dvx[:, :] = np.nan
        N = np.shape(VEL)[1]

        for i in range(2, N + 1):  # go through z co-ords (height)
            # print(i,N-i,N-i+1)
            Vx[:, N - i] = (VEL[:, N - i] - Vz[:, N - i + 1] * z[:, N - i]) / x[
                :, N - i
            ]  # calc Vx using Doppler vel eqn in cart. co-ords
            Vx[:, N - i][
                ZH[:, N - i] < Zedge
            ] = np.nan  # mask Vx below reflectivity threshold
            for j in range(1, np.shape(VEL)[0] - 1):  # go through x (range)
                dVx = np.mean(
                    [Vx[j + 1, N - i] - Vx[j, N - i], Vx[j, N - i] - Vx[j - 1, N - i]]
                )  # calc. divergence using mean of differences
                dvx[j, N - i] = dVx
                if np.isfinite(dVx):  # only calc Vz if we have a value for dVx
                    Vz[j, N - i] = (rho[N - i + 1] / rho[N - i]) * Vz[j, N - i + 1] + (
                        ((rho[N - i + 1] + rho[N - i]) / (2.0 * rho[N - i]))
                        * (dz / dx)
                        * dVx
                    )

        # double check fields for cloud edges with Zedge threshold are filtered out
        edges = np.where(ZH < Zedge)
        Vz[edges] = np.nan
        Vx[edges] = np.nan
        dvx[edges] = np.nan
        Vz[np.where(np.isnan(VEL))] = (
            np.nan
        )  # where VEL are nans fill Vz with nans to remove 0's
        # mask nans
        Vz = np.ma.array(Vz, mask=np.isnan(Vz))
        Vx = np.ma.array(Vx, mask=np.isnan(Vx))
        dvx = np.ma.array(dvx, mask=np.isnan(dvx))

        self.top_down_Vz = Vz
        self.top_down_Vx = Vx
        self.top_down_dvx = dvx
        # return Vz, Vx, dvx

    def nicol_weight_rhi(self, w_u, w_d, rho):
        """
        Nicol et al. (2015) error-weighting for RHI retrieval.

        Weight the ground up and top down retrievals to obtain a final retrieval weighted by the estimated error propagation
        due to density profile.

        :param w_u: w from ground up
        :param w_d: w from top down
        :param rho: density profile
        :return: Nicol weighted w
        """

        # intialise arrays and set up initial errors
        N_d = len(rho)  # np.shape(w_d)[1]
        # print(np.shape(new_dudr), np.shape(w_u), np.shape(w_d), len(rho))
        err_up = np.zeros(N_d)
        err_down = np.zeros(N_d)
        err_0 = 0.1  # Initial error. Confirmed by John Nicol by email
        err_up[0] = err_0
        err_down[-1] = err_0  #

        # error upwards
        for i in range(1, N_d):
            # print(i-1, i)
            err_up[i] = np.sqrt(((err_up[i - 1] * rho[i - 1]) / rho[i]) ** 2 + err_0**2)

        # error downwards
        for i in range(N_d - 2, -1, -1):
            # print i, i+1
            err_down[i] = np.sqrt(
                ((err_down[i + 1] * rho[i + 1]) / rho[i]) ** 2 + err_0**2
            )

        err_avg = (err_down * err_up) / np.sqrt(err_down**2 + err_up**2)

        weight_up = err_down**2 / (err_down**2 + err_up**2)
        weight_down = err_up**2 / (err_down**2 + err_up**2)

        # Nicol weight our integration

        nicol_w = w_u * weight_up + w_d * weight_down

        self.nicol_w = nicol_w
        # return nicol_w

    def main_procedure(self):
        # Variables to discuss
        Z_thresh = 0.0
        dx = 0.500  # Guessed these values by reverse working the Figure 5 plot within Nicol et al. (2015)
        dz = 0.250
        z_min = 0
        z_max = 12
        fdx = 0.050  # Values from John Nicol's MATLAB script
        fdz = 0.050

        # z-profile for rho calculation
        z_grid = np.arange(z_min, z_max, dz)
        # z_points = (z_edges + dz / 2)[:-1]

        self.calc_rVr(
            Zh=self.camra_scan.dbzh,
            Vr=self.camra_scan.vel_hv,
            r=self.camra_scan.range,
            Z_thresh=Z_thresh,
        )
        self.xx, self.zz, self.regrid_Z, self.xx_edges, self.zz_edges = self.grid_rhi(
            data=self.camra_scan.dbzh,
            r=self.camra_scan.r_coords / 1000,
            z=self.camra_scan.z_coords / 1000,
            dx=dx,
            dz=dz,
            fdx=fdx,
            fdz=fdz,
            xmin=0.0,
            xmax=180.0,
            zmin=0.0,
            zmax=12.0,
            method="linear",
            fill_value=np.nan,
        )
        _, _, self.regrid_rVr, _, _ = self.grid_rhi(
            data=self.rVr,
            r=self.camra_scan.r_coords / 1000,
            z=self.camra_scan.z_coords / 1000,
            dx=dx,
            dz=dz,
            fdx=fdx,
            fdz=fdz,
            xmin=0.0,
            xmax=180.0,
            zmin=0.0,
            zmax=12.0,
            method="linear",
            fill_value=np.nan,
        )
        self.calc_rho_profile(z_list=z_grid, mean_dz=dz * 1000)
        self.ground_up_vx_rhi(
            ZH=self.regrid_Z,
            VEL=self.regrid_rVr,
            x=self.xx,
            z=self.zz,
            dx=dx,
            dz=dz,
            rho=self.rho_profile,
            Zedge=Z_thresh,
        )
        self.top_down_vx_rhi(
            ZH=self.regrid_Z,
            VEL=self.regrid_rVr,
            x=self.xx,
            z=self.zz,
            dx=dx,
            dz=dz,
            rho=self.rho_profile,
            Zedge=Z_thresh,
        )
        self.nicol_weight_rhi(
            w_u=self.ground_up_Vz, w_d=self.top_down_Vz, rho=self.rho_profile
        )

    def plot_nicol_w(self):
        fig, ax = plt.subplots()
        pcm = ax.pcolormesh(
            self.xx_edges,
            self.zz_edges,
            self.nicol_w,
            cmap="seismic",
            vmin=-10,
            vmax=10,
        )
        cbar = plt.colorbar(pcm)
        cbar.set_label("Retrieved w / ms$^{-1}$")
        # plt.show()
        plt.savefig("test.png")


if __name__ == "__main__":
    cl = CLP(
        file="../../../data/ncas-radar-camra-1_cao_20230612-130655_rhi_l1_v1.0.0.nc"
    )
    cu = CamraUpdrafts(camra_obj=cl)
    cu.main_procedure()
    cu.plot_nicol_w()
