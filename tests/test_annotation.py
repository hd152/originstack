"""Tests for src/annotation.py (--annotate).

Network calls (SIMBAD TAP) are mocked at the urlopen layer, same pattern as
tests/test_net_query.py. WCS math is checked against astropy's own WCS
object (not reimplemented independently), so these tests validate this
module's usage of it, not astropy itself.
"""
from __future__ import annotations

import json
from unittest import mock

import numpy as np
import pytest
from astropy.io import fits

from src import annotation as an


def _fake_response(data: bytes):
    m = mock.MagicMock()
    m.read.return_value = data
    m.__enter__.return_value = m
    m.__exit__.return_value = False
    return m


def _wcs_header(ra=270.62, dec=-22.97, w=800, h=600, scale_deg_per_px=0.0004):
    h_ = fits.Header()
    h_['NAXIS'] = 3
    h_['NAXIS1'] = w
    h_['NAXIS2'] = h
    h_['NAXIS3'] = 3
    h_['CTYPE1'] = 'RA---TAN'
    h_['CTYPE2'] = 'DEC--TAN'
    h_['CRVAL1'] = ra
    h_['CRVAL2'] = dec
    h_['CRPIX1'] = w / 2.0
    h_['CRPIX2'] = h / 2.0
    h_['CD1_1'] = -scale_deg_per_px
    h_['CD1_2'] = 0.0
    h_['CD2_1'] = 0.0
    h_['CD2_2'] = scale_deg_per_px
    return h_


class TestBuildWcs:
    def test_returns_none_without_wcs_keys(self):
        h = fits.Header()
        h['NAXIS'] = 2
        assert an._build_wcs(h) is None

    def test_builds_2d_celestial_wcs(self):
        w = an._build_wcs(_wcs_header())
        assert w is not None
        assert w.has_celestial
        # round-trip: center pixel -> world -> back to center pixel
        ra, dec = w.all_pix2world([[400, 300]], 0)[0]
        px = w.all_world2pix([[ra, dec]], 0)[0]
        np.testing.assert_allclose(px, [400, 300], atol=1e-6)


class TestFieldCenterAndRadius:
    def test_center_matches_crval(self):
        w = an._build_wcs(_wcs_header(ra=270.62, dec=-22.97, w=800, h=600))
        ra_c, dec_c, radius = an._field_center_and_radius(w, (600, 800))
        assert ra_c == pytest.approx(270.62, abs=1e-3)
        assert dec_c == pytest.approx(-22.97, abs=1e-3)
        assert radius > 0

    def test_radius_scales_with_frame_size(self):
        w_small = an._build_wcs(_wcs_header(w=200, h=150))
        w_big = an._build_wcs(_wcs_header(w=2000, h=1500))
        _, _, r_small = an._field_center_and_radius(w_small, (150, 200))
        _, _, r_big = an._field_center_and_radius(w_big, (1500, 2000))
        assert r_big > r_small


class TestQueryAnnotationObjects:
    def test_combines_star_and_dso_queries(self):
        star_payload = {
            "metadata": [{"name": "ra"}, {"name": "dec"}, {"name": "main_id"},
                        {"name": "otype"}, {"name": "vmag"}],
            "data": [[270.6, -23.0, "HD 164492", "*", 6.8]],
        }
        dso_payload = {
            "metadata": [{"name": "ra"}, {"name": "dec"}, {"name": "main_id"},
                        {"name": "otype"}],
            "data": [[270.62, -23.03, "M  20", "OpC"]],
        }
        responses = [_fake_response(json.dumps(star_payload).encode()),
                    _fake_response(json.dumps(dso_payload).encode())]
        with mock.patch("urllib.request.urlopen", side_effect=responses):
            objects = an.query_annotation_objects(270.6, -23.0, 0.5)
        assert len(objects) == 2
        kinds = {o['kind'] for o in objects}
        assert kinds == {'star', 'dso'}
        star = next(o for o in objects if o['kind'] == 'star')
        assert star['name'] == 'HD 164492'
        dso = next(o for o in objects if o['kind'] == 'dso')
        assert dso['name'] == 'M  20'

    def test_query_failure_returns_empty_not_raises(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("network down")):
            objects = an.query_annotation_objects(270.6, -23.0, 0.5)
        assert objects == []

    def test_one_query_failing_still_returns_the_other(self):
        star_payload = {
            "metadata": [{"name": "ra"}, {"name": "dec"}, {"name": "main_id"},
                        {"name": "otype"}, {"name": "vmag"}],
            "data": [[270.6, -23.0, "HD 164492", "*", 6.8]],
        }
        responses = [_fake_response(json.dumps(star_payload).encode()),
                    OSError("network down")]
        with mock.patch("urllib.request.urlopen", side_effect=responses):
            objects = an.query_annotation_objects(270.6, -23.0, 0.5)
        assert len(objects) == 1
        assert objects[0]['kind'] == 'star'


class TestDrawAnnotations:
    def test_returns_none_without_pillow(self, monkeypatch):
        monkeypatch.setattr(an, 'Image', None)
        result = an.draw_annotations(np.zeros((10, 10, 3), np.uint8), None, [{'ra': 0, 'dec': 0, 'name': 'x', 'kind': 'star'}])
        assert result is None

    def test_empty_objects_returns_input_unchanged(self):
        preview = (np.random.rand(20, 20, 3) * 255).astype(np.uint8)
        result = an.draw_annotations(preview, None, [])
        assert np.array_equal(result, preview)

    def test_draws_visible_marks_for_in_frame_object(self):
        w = an._build_wcs(_wcs_header(w=200, h=150))
        preview = np.zeros((150, 200, 3), np.uint8)
        ra, dec = w.all_pix2world([[100, 75]], 0)[0]
        objects = [{'ra': ra, 'dec': dec, 'name': 'Test Star', 'otype': '*', 'kind': 'star'}]
        result = an.draw_annotations(preview, w, objects)
        assert result is not None
        assert result.shape == preview.shape
        assert not np.array_equal(result, preview)  # something got drawn

    def test_skips_object_projecting_outside_frame(self):
        w = an._build_wcs(_wcs_header(w=200, h=150))
        preview = np.zeros((150, 200, 3), np.uint8)
        # Far outside the field -- projects way off-frame.
        objects = [{'ra': 0.0, 'dec': 89.0, 'name': 'Nowhere', 'otype': '*', 'kind': 'star'}]
        result = an.draw_annotations(preview, w, objects)
        assert np.array_equal(result, preview)  # nothing drawn, unchanged


class TestRunAnnotation:
    def test_no_wcs_returns_false(self, tmp_path):
        header = fits.Header()
        header['NAXIS'] = 2
        args = mock.Mock(annotate_mag_limit=9.0, stretch='ghs', ghs_b=5.0,
                         ghs_sp=0.10, ghs_hp=0.95, preview_black_sigma=-0.5)
        stacked = np.zeros((50, 50, 3), np.float32)
        ok = an.run_annotation(stacked, header, str(tmp_path / "out.fits"), args)
        assert ok is False

    def test_no_objects_found_returns_false(self, tmp_path):
        header = _wcs_header(w=50, h=50)
        args = mock.Mock(annotate_mag_limit=9.0, stretch='ghs', ghs_b=5.0,
                         ghs_sp=0.10, ghs_hp=0.95, preview_black_sigma=-0.5)
        stacked = np.full((50, 50, 3), 1000.0, np.float32)
        with mock.patch.object(an, 'query_annotation_objects', return_value=[]):
            ok = an.run_annotation(stacked, header, str(tmp_path / "out.fits"), args)
        assert ok is False

    def test_writes_annotated_jpg_on_success(self, tmp_path):
        header = _wcs_header(w=50, h=50)
        args = mock.Mock(annotate_mag_limit=9.0, stretch='ghs', ghs_b=5.0,
                         ghs_sp=0.10, ghs_hp=0.95, preview_black_sigma=-0.5)
        rng = np.random.default_rng(0)
        stacked = 1000.0 + rng.standard_normal((50, 50, 3)).astype(np.float32) * 30.0
        w = an._build_wcs(header)
        ra, dec = w.all_pix2world([[25, 25]], 0)[0]
        fake_objects = [{'ra': ra, 'dec': dec, 'name': 'Test', 'otype': '*', 'kind': 'star'}]
        out_path = str(tmp_path / "out.fits")
        with mock.patch.object(an, 'query_annotation_objects', return_value=fake_objects):
            ok = an.run_annotation(stacked, header, out_path, args)
        assert ok is True
        assert (tmp_path / "out_annotated.jpg").exists()
