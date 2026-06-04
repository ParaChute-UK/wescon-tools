import numpy as np
import xarray as xr
import pytest
from wescon_radar_dev import CompareDeltaZCandidates


def make_label_arrays():
    """
    Two label arrays with a known overlap pattern:
      - Cloud 1 in labels1 overlaps with Cloud 1 in labels2
      - Cloud 2 in labels2 does NOT overlap with anything in labels1
    Grid is 10x20 (z x x).
    """
    nz, nx = 10, 20
    labels1 = np.zeros((nz, nx), dtype=int)
    labels2 = np.zeros((nz, nx), dtype=int)

    labels1[:, 5:10] = 1
    labels2[:, 5:10] = 1   # overlap with cloud 1

    labels2[:, 14:18] = 2  # only in labels2

    return labels1, labels2


def make_minimal_obj_dataset(cloud_labels, cloud_max_z=5.0):
    """Build the minimal xr.Dataset that find_overlapping_cloud_matches expects."""
    unique = [l for l in np.unique(cloud_labels) if l != 0]
    n_clouds = len(unique)
    ds = xr.Dataset(
        data_vars=dict(
            cloud_label=(['time', 'cloud_id'], [unique]),
            cloud_max_z=(['time', 'reflectivity_thresh', 'cloud_id'],
                         [[[cloud_max_z] * n_clouds]]),
        ),
        coords=dict(
            time=[0],
            cloud_id=np.arange(n_clouds),
            reflectivity_thresh=[10],
        ),
    )
    return ds


class TestFindOverlappingCloudMatches:

    def test_overlapping_clouds_are_matched(self):
        labels1, labels2 = make_label_arrays()
        objs1 = make_minimal_obj_dataset([1])
        objs2 = make_minimal_obj_dataset([1, 2])
        matches = CompareDeltaZCandidates.find_overlapping_cloud_matches(
            labels1, labels2, objs1, objs2
        )
        assert (1, 1) in matches

    def test_non_overlapping_cloud_not_matched(self):
        labels1, labels2 = make_label_arrays()
        objs1 = make_minimal_obj_dataset([1])
        objs2 = make_minimal_obj_dataset([1, 2])
        matches = CompareDeltaZCandidates.find_overlapping_cloud_matches(
            labels1, labels2, objs1, objs2
        )
        assert (1, 2) not in matches

    def test_no_overlap_returns_empty(self):
        labels1 = np.zeros((10, 20), dtype=int)
        labels2 = np.zeros((10, 20), dtype=int)
        labels1[:, :5] = 1    # left side
        labels2[:, 15:] = 1   # right side, no overlap
        objs1 = make_minimal_obj_dataset([1])
        objs2 = make_minimal_obj_dataset([1])
        matches = CompareDeltaZCandidates.find_overlapping_cloud_matches(
            labels1, labels2, objs1, objs2
        )
        assert matches == []
