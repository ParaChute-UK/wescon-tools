import cartopy.crs as ccrs
import shapely.geometry as sgeom


class CustomOSGB(ccrs.TransverseMercator):
    """Copied from cartopy but with different y_limits.

    Stops coastlines/contourf etc. stopping near N France."""
    def __init__(self):
        super().__init__(central_longitude=-2, central_latitude=49,
                         scale_factor=0.9996012717,
                         false_easting=400000, false_northing=-100000,
                         globe=ccrs.Globe(datum='OSGB36', ellipse='airy'))

    @property
    def boundary(self):
        x0, x1 = self.x_limits
        y0, y1 = self.y_limits
        return sgeom.LineString([(x0, y0), (x0, y1),
                                 (x1, y1), (x1, y0),
                                 (x0, y0)])

    @property
    def x_limits(self):
        return (-2e7, 2e7)

    @property
    def y_limits(self):
        return (-1e7, 1e7)

