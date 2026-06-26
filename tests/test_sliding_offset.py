import numpy as np
import pytest
from delta_z import sliding_offset_to_slices, find_sliding_min_rmse


class TestSlidingOffsetToSlices:
    """Pin the slice behaviour for every valid offset in [-3, 3]."""

    @pytest.mark.parametrize('idx,expected_a1,expected_a2', [
        (-3, [0],          [3]),
        (-2, [0, 1],       [2, 3]),
        (-1, [0, 1, 2],    [1, 2, 3]),
        ( 0, [0, 1, 2, 3], [0, 1, 2, 3]),
        ( 1, [1, 2, 3],    [0, 1, 2]),
        ( 2, [2, 3],       [0, 1]),
        ( 3, [3],          [0]),
    ])
    def test_slices(self, idx, expected_a1, expected_a2):
        a = np.array([0, 1, 2, 3])
        s1, s2 = sliding_offset_to_slices(idx)
        assert list(a[s1]) == expected_a1
        assert list(a[s2]) == expected_a2

    def test_zero_offset_returns_full_arrays(self):
        a = np.arange(4)
        s1, s2 = sliding_offset_to_slices(0)
        np.testing.assert_array_equal(a[s1], a)
        np.testing.assert_array_equal(a[s2], a)


class TestFindSlidingMinRmse:
    """Known-answer tests that pin the sign convention of the returned offset."""

    def test_identical_arrays_return_zero(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        assert find_sliding_min_rmse(a, a) == 0

    def test_a2_shifted_right_by_one(self):
        # a1=[1,2,3,4], a2=[2,3,4,5]: at offset=+1 a1[1:]==[2,3,4] vs a2[:-1]==[2,3,4] → RMSE=0
        a1 = np.array([1.0, 2.0, 3.0, 4.0])
        a2 = np.array([2.0, 3.0, 4.0, 5.0])
        assert find_sliding_min_rmse(a1, a2) == 1

    def test_a2_shifted_left_by_one(self):
        # a1=[2,3,4,5], a2=[1,2,3,4]: at offset=-1 a1[:-1]==[2,3,4] vs a2[1:]==[2,3,4] → RMSE=0
        a1 = np.array([2.0, 3.0, 4.0, 5.0])
        a2 = np.array([1.0, 2.0, 3.0, 4.0])
        assert find_sliding_min_rmse(a1, a2) == -1

    def test_a2_shifted_right_by_two(self):
        a1 = np.array([1.0, 2.0, 3.0, 4.0])
        a2 = np.array([3.0, 4.0, 5.0, 6.0])
        assert find_sliding_min_rmse(a1, a2) == 2

    def test_requires_length_4(self):
        with pytest.raises(AssertionError):
            find_sliding_min_rmse(np.array([1, 2, 3]), np.array([1, 2, 3]))
