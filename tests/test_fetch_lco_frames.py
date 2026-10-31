"""tools/fetch_lco_frames.py: frame selection and the query (no network)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
import fetch_lco_frames as lco  # noqa: E402


def _f(tel, inst, filt, day, i=0):
    return {'telescope_id': tel, 'instrument_id': inst, 'primary_optical_element': filt,
            'observation_day': day, 'observation_date': f'{day}T00:00:{i:02d}Z'}


def test_select_takes_the_biggest_group_and_whole_nights_first():
    frames = ([_f('0m4a', 'sq31', 'rp', '20260101', i) for i in range(2)]
              + [_f('0m4a', 'sq31', 'rp', '20260102', i) for i in range(5)]
              + [_f('0m4b', 'sq33', 'V', '20260103', i) for i in range(4)]
              + [_f('1m0a', 'fa05', 'rp', '20260104')])
    pick = lco.select(frames, '0m4', None, 6)
    assert {f['instrument_id'] for f in pick} == {'sq31'}          # 7 frames beat sq33's 4
    assert [f['observation_day'] for f in pick] == ['20260102'] * 5 + ['20260101']


def test_select_filters_telescope_class_and_filter_and_handles_none():
    frames = [_f('0m4a', 'sq31', 'rp', 'd1'), _f('1m0a', 'fa05', 'gp', 'd1'), _f('1m0a', 'fa05', 'rp', 'd2')]
    assert [f['primary_optical_element'] for f in lco.select(frames, '1m0', 'rp', 9)] == ['rp']
    assert lco.select(frames, '2m0', None, 9) == []
    assert lco.select([], None, None, 9) == []


def test_query_url_asks_for_public_reduced_science_frames():
    u = lco.query_url('M 13', 10, 600, '2026-01-01', None)
    assert 'public=true' in u and 'reduction_level=91' in u and 'target_name=M+13' in u
    assert 'start=2026-01-01' in u and 'end=' not in u
