import warnings

import numpy as np
import skimage as ski


def _set_to_local_mean(fmod, fin, i, j):
    """Modify fmod in place be replacing value at i, j with local mean

    local mean being a mean over the surrounding 9 cells.
    """
    istart = i - 1
    iend = i + 2
    jstart = j - 1
    jend = j + 2
    if istart < 0:
        istart = 0
    if jstart < 0:
        jstart = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # Don't want to be told if all the values are nan.
        fmod[i, j] = np.nanmean(fin[istart:iend, jstart:jend])

def fill_nan_local_mean(f):
    """Take a field and fill with local mean (surrounding 3x3 cells).

    :param f: input 2d field (numpy array)
    :returns: output 2d field with all nans filled in.
    """
    assert not np.isnan(f).all(), 'input field is entirely nans'
    fout = f.copy()

    while np.isnan(fout).any():
        ftemp = fout.copy()
        for i, j in zip(*np.where(np.isnan(fout))):
            # print(i, j)
            _set_to_local_mean(ftemp, fout, i, j)
        fout = ftemp

    return fout


class FlowInterp:
    """Interpolation of two 2D numpy fields based on optical flow and image warping.

    * Calculate optical flow between 2 snapshots
    * Apply physical constraints to vector (e.g., no winds greater than 50 m/s)
    * Use flow field to warp/morph first field into second based on frac (0-1)
    * Simultaneously do standard interpolation between first and second field
    """
    def __init__(self, f1, f2, subsample='stride', stride=10, nrand=1000, dx=1000, dt=300, apply_physical_constraints=True, max_flow_speed=40, optical_flow_kwargs=None):
        """Initialize with two input fields, and a stride length for sampling for later warp.

        :param f1: input field 1 (2d np.array)
        :param f2: input field 2 (2d np.array)
        :param stride: how regularly to sample input fields for warp
        :param dx: grid spacing in m
        :param dt: time between f1 and f2 in s
        :param apply_physical_constraints: replace large flow vectors with local avg
        :param max_flow_speed: limit on flow speed for physical constraint
        :returns: None
        """
        self.f1 = f1
        self.f2 = f2
        self.stride = stride
        self.dx = dx
        self.dt = dt
        self.max_flow_speed = max_flow_speed
        if optical_flow_kwargs is None:
            # These values were chosen by minimizing the RMSE between two fields 10 mins apart
            # interp'd to 5 mins and then RMSE with actual field at 5 mins.
            optical_flow_kwargs = {'attachment': 1, 'tightness': 0.3}

        # Calculate optical flow.
        self.flow_vec = ski.registration.optical_flow_tvl1(f1, f2, **optical_flow_kwargs)
        if apply_physical_constraints:
            # Apply physical constraints.
            self._apply_physical_constraints()

        # Note these are all in grid coords.
        x = np.arange(f1.shape[1])
        y = np.arange(f1.shape[0])
        X, Y = np.meshgrid(x, y)

        if subsample == 'stride':
            # Set up x/y points for warp.
            self.xpoints = X[::stride, ::stride].flatten()
            self.ypoints = Y[::stride, ::stride].flatten()
            # 1 *is* index for x!
            self.xflow_vec = self.flow_vec[1, ::stride, ::stride].flatten()
            self.yflow_vec = self.flow_vec[0, ::stride, ::stride].flatten()
        elif subsample == 'random':
            raise NotImplementedError('subsample == "random" not working')
            # Not currently working.
            # Perhaps because it doesn't have a boundary?
            class Point:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
            points = list(Point(x, y) for x, y in zip(X.flatten(), Y.flatten()))
            #print(xxyy.shape)
            d = self.f1.copy()
            d[d > 20] = 20
            d = d + 0.1
            choices = np.random.choice(points, nrand, p=(d / d.sum()).flatten())
            rpoints = np.array([(p.x, p.y) for p in choices])
            self.xpoints = rpoints[:, 0]
            self.ypoints = rpoints[:, 1]
            # plt.scatter(self.xpoints, self.ypoints)
            self.xflow_vec = self.flow_vec[1][(self.ypoints, self.xpoints)]
            self.yflow_vec = self.flow_vec[0][(self.ypoints, self.xpoints)]

    def _apply_physical_constraints(self):
        # TODO: would be better to to this as a multiple of the grid-mean flow speed?
        constraint = (self.flow_vec[1]**2 + self.flow_vec[0]**2) * self.dx / self.dt > self.max_flow_speed**2
        self.flow_vec[1][constraint] = np.nan
        self.flow_vec[0][constraint] = np.nan
        self.flow_vec[1] = fill_nan_local_mean(self.flow_vec[1])
        self.flow_vec[0] = fill_nan_local_mean(self.flow_vec[0])

    def interp(self, frac, warp=True, temporal_interp=True, f1=None, f2=None):
        """Perform interpolation.

        :param frac: fraction between field 1 and 2 (0-1)
        :returns: flow-interpolated field
        """
        assert warp or temporal_interp, 'One of warp and temporal_interp must be True'
        if f1 is None:
            f1 = self.f1
        if f2 is None:
            f2 = self.f2

        src = np.array(list(zip(self.xpoints, self.ypoints)))

        if warp:
            tps = ski.transform.ThinPlateSplineTransform()
            dst = np.array(list(zip(self.xpoints + frac * self.xflow_vec, self.ypoints + frac * self.yflow_vec)))
            tps.estimate(dst, src)
            warped_f1 = ski.transform.warp(f1, tps)

            if temporal_interp:
                tps = ski.transform.ThinPlateSplineTransform()
                dst = np.array(list(zip(self.xpoints - (1 - frac) * self.xflow_vec, self.ypoints - (1 - frac) * self.yflow_vec)))
                tps.estimate(dst, src)
                warped_f2 = ski.transform.warp(f2, tps)
        else:
            warped_f1 = f1
            warped_f2 = f2

        if temporal_interp:
            warped = frac * warped_f2 + (1 - frac) * warped_f1
        else:
            warped = warped_f1
        return warped
