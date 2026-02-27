import sys
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from shapely import box, Point
from shapely.geometry import MultiPoint

from wescon_tools.custom_osgb import CustomOSGB
from wescon_tools.match_rhi_to_3d_winds import MatchRHIto3dWinds, Plot3dWinds, CHIL_X, CHIL_Y

import numpy as np
from shapely.geometry import Point, Polygon


def create_sector(center_x, center_y, radius, start_angle, end_angle):
    """
    Creates a sector polygon.
    Angles are in degrees, clockwise from North (0°).
    """
    # Create the circle center point
    center = Point(center_x, center_y)

    # Generate points along the arc
    # Note: Math functions use radians and 0 at East (3 o'clock)
    # We convert 'North-based degrees' to 'Math-based radians'
    t = np.linspace(np.deg2rad(90 - start_angle),
                    np.deg2rad(90 - end_angle), 100)

    arc_points = [(center_x + radius * np.cos(angle),
                   center_y + radius * np.sin(angle)) for angle in t]

    # Build the polygon: [Center] + [Points along the arc] + [Center]
    return Polygon([center] + arc_points + [center])


def plot_3d_winds_CAMRa_domain(matcher):
    # Low-stakes, can verify it works. Pretty much vibe coded.
    proj = CustomOSGB()
    ds3d_osgb = matcher.ds3d_osgb

    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=proj), layout='constrained')
    ax.coastlines()

    # 1. Define Rectangle Geometry
    minx = ds3d_osgb.eastings.min().values.item()
    maxx = ds3d_osgb.eastings.max().values.item()
    miny = ds3d_osgb.northings.min().values.item()
    maxy = ds3d_osgb.northings.max().values.item()

    rect_geom = box(minx, miny, maxx, maxy)

    # 2. Define Circle Geometry (150km radius)
    circle_geom = Point(CHIL_X, CHIL_Y).buffer(150e3)

    # 3. Calculate Intersection and Area
    intersection_geom = rect_geom.intersection(circle_geom)
    intersection_area_km2 = intersection_geom.area / 1e6
    circle_area_km2 = circle_geom.area / 1e6

    # 4. Plotting using Cartopy's add_geometries
    # This replaces the need for mpatches.Rectangle/Circle
    ax.add_geometries([rect_geom], crs=proj, facecolor='red', alpha=0.3, edgecolor='red')
    ax.add_geometries([circle_geom], crs=proj, facecolor='blue', alpha=0.2, edgecolor='blue')
    ax.add_geometries([intersection_geom], crs=proj, facecolor='yellow', alpha=0.5, edgecolor='black')
    ax.plot(CHIL_X, CHIL_Y, 'ko', transform=proj)

    # 1. Get the boundary (perimeter) of each shape
    rect_perimeter = rect_geom.boundary
    circle_perimeter = circle_geom.boundary

    # 2. Find where the two perimeters meet
    intersection_points = rect_perimeter.intersection(circle_perimeter)

    # 3. Extract and print coordinates
    if isinstance(intersection_points, MultiPoint):
        for i, pt in enumerate(intersection_points.geoms):
            print(f"Intersection {i + 1}: Easting={pt.x:.2f}, Northing={pt.y:.2f}")
    elif not intersection_points.is_empty:
        print(f"Intersection: Easting={intersection_points.x:.2f}, Northing={intersection_points.y:.2f}")

    # 4. Optional: Plot the points as red dots
    ax.plot([p.x for p in intersection_points.geoms],
            [p.y for p in intersection_points.geoms],
            'ro', transform=proj, markersize=8)

    def get_sector_angles(geoms):
        assert len(geoms) == 2
        angles = []
        for i, pt in enumerate(geoms):
            dx = pt.x - CHIL_X
            dy = pt.y - CHIL_Y

            heading_rad = np.arctan2(dx, dy)
            heading_deg = np.rad2deg(heading_rad)

            # Normalize to 0-360 range
            heading_final = heading_deg % 360
            print(f"Point {i + 1} Heading: {heading_final:.2f}°")
            angles.append(heading_final)
        return sorted(angles)

    sector_angles = get_sector_angles(intersection_points.geoms)

    sector_geom = create_sector(CHIL_X, CHIL_Y, 150e3, sector_angles[0], sector_angles[1])

    ax.add_geometries([sector_geom], crs=proj, facecolor='green', alpha=0.4)

    ax.set_extent([CHIL_X - 200e3, CHIL_X + 200e3, CHIL_Y - 200e3, CHIL_Y + 200e3], crs=proj)
    ax.set_title(
        f"Intersection Area: {intersection_area_km2:.2f} km$^2$\n"
        f"Percentage of CAMRa coverage: {intersection_area_km2 / circle_area_km2 * 100:.2f}%\n"
        f"Full beam covered: {sector_angles[0]:.2f}° - {sector_angles[1]:.2f}° "
        f"({(sector_angles[1] - sector_angles[0]) / 360 * 100:.2f}%)")
    plt.show()


if __name__ == '__main__':
    figdir = Path(f'/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_figs/3D_winds/v6.1/')
    figdir.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        camra_path = Path(sys.argv[1])
        time_interp = sys.argv[2] == 'True'
    else:
        # defaults
        # /gws/pw/j07/woest/rjthomps/winds3d/data/20230803/grid_1000m_filter_1_0_20230803_1310_v6.1.nc
        camra_path = Path(
            '/gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_output/wescon_radar_dev/v7/20230803/camra'
            '/gridded_ncas-radar-camra-1_cao_20230803-131042_rhi_l1_v1.0.0.nc')
        time_interp = True

    matcher = locals().get('matcher', None)
    prev_argv = locals().get('prev_argv', None)
    if matcher is None or prev_argv != sys.argv:
        print('Load objs')
        # Close in time to the RHI scan I want to compare with.
        ds_rad = xr.open_dataset(camra_path).sel(time='2023-08-03 13:10:42', method='nearest')
        matcher = MatchRHIto3dWinds(ds_rad)
        matcher.match()
        prev_argv = sys.argv

    plot_3d_winds_CAMRa_domain(matcher)
    outfilepath = figdir / f'3D_winds_CAMRA_domains.png'
    plt.savefig(outfilepath)

    print('plot data')
    plotter = Plot3dWinds(matcher)
    plotter.plot()

    camra_time = pd.Timestamp(matcher.ds_rad.time.values.item())
    tstr = camra_time.strftime('%H%M%S')
    interpstr = 'interp' if matcher.time_interp else 'nearest'
    outfilepath = figdir / f'wescon.{camra_path.stem}.CAMRa_{tstr}.{interpstr}.png'
    print(outfilepath)
    plt.savefig(outfilepath)
