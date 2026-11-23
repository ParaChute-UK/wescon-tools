import pandas as pd
import pytest
from wescon_radar_dev import find_brackets
from helpers import make_bracket_df


def two_complete_brackets():
    """4 scans + 4 scans, separated by a large az jump."""
    azs = [180.0, 180.2, 180.4, 180.6,   # bracket A
           181.5, 181.7, 181.9, 182.1]    # bracket B (large gap resets)
    return make_bracket_df(azs)


def incomplete_then_complete():
    """3 scans (incomplete) then 4 scans (complete)."""
    azs = [180.0, 180.2, 180.4,           # incomplete bracket
           181.5, 181.7, 181.9, 182.1]    # complete bracket
    return make_bracket_df(azs)


class TestFindBrackets:

    def test_two_complete_brackets_assigned_different_ids(self):
        df = two_complete_brackets()
        find_brackets(df)
        assert df.iloc[0]['bracket'] != df.iloc[4]['bracket']

    def test_scans_within_bracket_share_bracket_id(self):
        df = two_complete_brackets()
        find_brackets(df)
        assert len(df[df.bracket == df.iloc[0]['bracket']]) == 4

    def test_bracket_idx_increments_within_bracket(self):
        df = two_complete_brackets()
        find_brackets(df)
        b0 = df[df.bracket == df.iloc[0]['bracket']]
        assert list(b0['bracket_idx']) == [0, 1, 2, 3]

    def test_large_az_gap_resets_bracket(self):
        df = two_complete_brackets()
        find_brackets(df)
        assert df.iloc[3]['bracket'] != df.iloc[4]['bracket']

    def test_incomplete_bracket_has_fewer_than_4_scans(self):
        df = incomplete_then_complete()
        find_brackets(df)
        first_bracket_id = df.iloc[0]['bracket']
        assert len(df[df.bracket == first_bracket_id]) == 3

    def test_single_scan_gets_its_own_bracket(self):
        df = make_bracket_df([180.0])
        find_brackets(df)
        assert len(df) == 1
        assert df.iloc[0]['bracket'] == 0

    def test_az_within_lower_limit_does_not_continue_bracket(self):
        # daz < 0.1 should NOT continue a bracket — treated as a new scan sequence.
        azs = [180.0, 180.05, 180.1, 180.15]
        df = make_bracket_df(azs)
        find_brackets(df)
        assert df.iloc[0]['bracket'] != df.iloc[1]['bracket']
