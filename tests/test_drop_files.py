"""Dropping a folder on the window (src/drop_files.py)."""
from src.drop_files import folder_from_drop


def test_a_dropped_folder_wins_over_files(tmp_path):
    (tmp_path / 'lights').mkdir()
    (tmp_path / 'a.fits').write_bytes(b'')
    assert folder_from_drop([str(tmp_path / 'a.fits'), str(tmp_path / 'lights')]) == str(tmp_path / 'lights')


def test_a_dropped_file_means_the_folder_holding_it(tmp_path):
    (tmp_path / 'Light0001.fits').write_bytes(b'')
    assert folder_from_drop([str(tmp_path / 'Light0001.fits')]) == str(tmp_path)


def test_nothing_usable_means_no_folder(tmp_path):
    assert folder_from_drop([]) is None
    assert folder_from_drop([str(tmp_path / 'gone.fits')]) is None
