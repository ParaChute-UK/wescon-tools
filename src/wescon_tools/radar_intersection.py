import dataclasses

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclasses.dataclass
class RadarIntersection:
    radar_name1: str
    radar_name2: str
    time1: pd.Timestamp
    az1: float
    x1: float
    y1: float
    u1: float
    v1: float
    time2: pd.Timestamp
    az2: float
    x2: float
    y2: float
    u2: float
    v2: float
    intersect_x: float
    intersect_y: float
    dist1: float
    dist2: float
    angle: float


def plot_radar_intersection(ri: RadarIntersection):
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.OSGB()})
    ax.plot(ri.x1, ri.y1, 'or')
    ax.plot([ri.x1, ri.x1 + ri.u1 * np.abs(ri.dist1)], [ri.y1, ri.y1 + ri.v1 * np.abs(ri.dist1)], '-r')

    ax.plot(ri.x2, ri.y2, 'ob')
    ax.plot([ri.x2, ri.x2 + ri.u2 * np.abs(ri.dist2)], [ri.y2, ri.y2 + ri.v2 * np.abs(ri.dist2)], '-b')

    ax.plot(ri.intersect_x, ri.intersect_y, marker='o', color='purple')
    ax.coastlines()


class RadarIntersectionCalculator:
    def __init__(self, radar_name1, radar_name2, x1, y1, x2, y2):
        self.radar_name1 = radar_name1
        self.radar_name2 = radar_name2
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def calc_intersect(self, time1, time2, az1, az2):
        u1 = np.sin(az1 * np.pi / 180)
        v1 = np.cos(az1 * np.pi / 180)
        u2 = np.sin(az2 * np.pi / 180)
        v2 = np.cos(az2 * np.pi / 180)

        M = np.matrix([[u1, -u2], [v1, -v2]])
        b = np.array([self.x2 - self.x1, self.y2 - self.y1])

        x = M.I @ b
        intersect_x = self.x1 + x[0, 0] * u1
        intersect_y = self.y1 + x[0, 0] * v1
        angle = (
            180 / np.pi * np.arccos(np.dot([u1, v1], [u2, v2]) / (np.linalg.norm([u1, v1]) * np.linalg.norm([u2, v2])))
        )
        return RadarIntersection(
            self.radar_name1,
            self.radar_name2,
            time1,
            az1,
            self.x1,
            self.y1,
            u1,
            v1,
            time2,
            az2,
            self.x2,
            self.y2,
            u2,
            v2,
            intersect_x,
            intersect_y,
            x[0, 0],
            x[0, 1],
            angle,
        )
