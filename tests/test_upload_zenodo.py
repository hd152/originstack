"""tools/upload_zenodo.py against a fake Zenodo: drafts, resume, verification, the publish guard."""
import hashlib
import importlib.util
import io
import json
import urllib.error
import urllib.parse
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    'upload_zenodo', Path(__file__).resolve().parents[1] / 'tools' / 'upload_zenodo.py')
uz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uz)


class _Resp(io.BytesIO):
    def __init__(self, obj):
        super().__init__(json.dumps(obj).encode())

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeZenodo:
    """Just enough of the deposit API: create, get, metadata, bucket PUT, files, publish."""

    def __init__(self):
        self.deps = {}
        self.calls = []
        self.next_id = 100

    def __call__(self, req, timeout=0):
        url, method = req.full_url, req.get_method()
        path = url.split('/api')[-1]
        self.calls.append((method, path))
        assert req.get_header('Authorization') == 'Bearer SECRET'
        body = req.data
        if method == 'POST' and path == '/deposit/depositions':
            self.next_id += 1
            d = {'id': self.next_id, 'submitted': False, 'files': {}, 'metadata': {},
                 'links': {'bucket': f'https://zenodo.org/api/files/b{self.next_id}',
                           'html': f'https://zenodo.org/deposit/{self.next_id}'}}
            self.deps[self.next_id] = d
            return _Resp(d)
        if path.startswith('/files/b'):
            bid, name = path[len('/files/b'):].split('/', 1)
            data = body.read() if hasattr(body, 'read') else body
            name = urllib.parse.unquote(name)
            self.deps[int(bid)]['files'][name] = {'filename': name, 'filesize': len(data),
                                                  'checksum': hashlib.md5(data).hexdigest()}
            return _Resp({'key': name})
        parts = path.split('/')
        dep = self.deps[int(parts[3])]
        if method == 'GET' and len(parts) == 4:
            return _Resp(dep)
        if method == 'PUT' and len(parts) == 4:
            if dep['submitted'] and not dep.get('editing'):
                raise urllib.error.HTTPError(url, 403, 'published', {}, io.BytesIO(b'locked'))
            dep['metadata'] = dict(json.loads(body)['metadata'], prereserve_doi={'doi': f'10.5281/zenodo.{dep["id"]}'})
            return _Resp(dep)
        if method == 'GET' and parts[4] == 'files':
            return _Resp(list(dep['files'].values()))
        if method == 'POST' and parts[-1] == 'edit':
            assert dep['submitted']
            dep['editing'] = True
            return _Resp(dep)
        if method == 'POST' and parts[-1] == 'publish':
            dep['submitted'] = True
            dep['editing'] = False
            return _Resp({'doi': f'10.5281/zenodo.{dep["id"]}', 'id': dep['id']})
        raise AssertionError((method, path))


META = {'title': 'T', 'upload_type': 'dataset', 'description': 'd',
        'creators': [{'name': 'OriginStack'}], 'license': 'cc-by-4.0'}


@pytest.fixture
def setup(tmp_path):
    f1, f2 = tmp_path / 'a.zip', tmp_path / 'b.zip'
    f1.write_bytes(b'x' * 1000)
    f2.write_bytes(b'y' * 500)
    rec = {'name': 'one', 'metadata': META, 'files': [str(f1), str(f2)]}
    fake = FakeZenodo()
    return rec, fake, uz.Zenodo('SECRET', opener=fake), str(tmp_path / 'm.json.state.json')


def test_manifest_check_finds_problems(tmp_path):
    good = tmp_path / 'f'
    good.write_bytes(b'1')
    assert uz.check_manifest({'records': [{'name': 'a', 'metadata': META, 'files': [str(good)]}]}) == []
    bad = uz.check_manifest({'records': [{'name': 'a', 'metadata': {'title': 'x'}, 'files': ['/nope']},
                                         {'name': 'a', 'metadata': META, 'files': []}]})
    text = ' | '.join(bad)
    assert 'repeated' in text and 'metadata.creators is missing' in text and 'file not found' in text
    assert 'no files' in text


def test_upload_creates_a_draft_and_verifies(setup):
    rec, fake, api, state_path = setup
    state = {}
    dep_id = uz.upload_record(api, rec, state, state_path, log=lambda *_: None)
    assert all(fake.deps[dep_id]['metadata'][k] == v for k, v in META.items())
    assert not fake.deps[dep_id]['submitted']
    assert set(fake.deps[dep_id]['files']) == {'a.zip', 'b.zip'}
    assert uz.verify_record(api, rec, dep_id) == []
    assert json.loads(Path(state_path).read_text())['one']['id'] == dep_id


def test_resume_skips_files_already_there(setup):
    rec, fake, api, state_path = setup
    state = {}
    uz.upload_record(api, rec, state, state_path, log=lambda *_: None)
    fake.calls.clear()
    uz.upload_record(api, rec, state, state_path, log=lambda *_: None)
    assert not [c for c in fake.calls if c[1].startswith('/files/')]               # no second upload
    assert ('POST', '/deposit/depositions') not in fake.calls                      # no second draft


def test_a_changed_file_is_uploaded_again_and_verify_catches_differences(setup):
    rec, fake, api, state_path = setup
    state = {}
    dep_id = uz.upload_record(api, rec, state, state_path, log=lambda *_: None)
    Path(rec['files'][0]).write_bytes(b'z' * 1000)                                  # same size, new bytes
    assert any('checksum' in p for p in uz.verify_record(api, rec, dep_id))
    uz.upload_record(api, rec, state, state_path, log=lambda *_: None)
    assert uz.verify_record(api, rec, dep_id) == []
    fake.deps[dep_id]['files']['stray.zip'] = {'filename': 'stray.zip', 'filesize': 1, 'checksum': 'x'}
    assert any('not in the manifest' in p for p in uz.verify_record(api, rec, dep_id))


def test_a_published_record_is_never_touched(setup):
    rec, fake, api, state_path = setup
    state = {}
    dep_id = uz.upload_record(api, rec, state, state_path, log=lambda *_: None)
    api.publish(dep_id)
    with pytest.raises(RuntimeError, match='already published'):
        uz.upload_record(api, rec, state, state_path, log=lambda *_: None)


def test_main_publishes_only_on_request_and_refuses_a_mismatch(setup, tmp_path, monkeypatch, capsys):
    rec, fake, api, state_path = setup
    man = tmp_path / 'm.json'
    man.write_text(json.dumps({'records': [rec]}))
    monkeypatch.setenv('ZENODO_TOKEN', 'SECRET')
    monkeypatch.setattr(uz.urllib.request, 'urlopen', fake)
    assert uz.main([str(man)]) == 0                                                  # drafts only
    dep_id = json.loads(Path(state_path).read_text())['one']['id']
    assert not fake.deps[dep_id]['submitted'] and 'nothing is published' in capsys.readouterr().out
    Path(rec['files'][1]).write_bytes(b'q' * 500)                                    # local file changed after upload
    with pytest.raises(SystemExit) as e:
        uz.main([str(man), '--publish'])
    assert 'refusing to publish' in str(e.value) and not fake.deps[dep_id]['submitted']
    assert uz.main([str(man)]) == 0                                                  # re-upload fixes the draft
    assert uz.main([str(man), '--publish']) == 0
    assert fake.deps[dep_id]['submitted']
    assert json.loads(Path(state_path).read_text())['one']['doi'].startswith('10.5281/')
    assert 'SECRET' not in capsys.readouterr().out


def test_retry_on_a_transient_error_then_success(monkeypatch):
    calls = {'n': 0}

    def flaky(req, timeout=0):
        calls['n'] += 1
        if calls['n'] < 3:
            raise urllib.error.HTTPError(req.full_url, 503, 'busy', {}, io.BytesIO(b'busy'))
        return _Resp({'ok': True})
    monkeypatch.setattr(uz.time, 'sleep', lambda s: None)
    assert uz.Zenodo('SECRET', opener=flaky).get(5) == {'ok': True} and calls['n'] == 3

    def hard(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 403, 'no', {}, io.BytesIO(b'forbidden'))
    with pytest.raises(RuntimeError, match='HTTP 403'):
        uz.Zenodo('SECRET', opener=hard).get(5)


def test_drafts_only_creates_drafts_with_reserved_dois_and_uploads_nothing(setup, tmp_path, monkeypatch, capsys):
    rec, fake, api, state_path = setup
    man = tmp_path / 'm.json'
    man.write_text(json.dumps({'records': [rec]}))
    monkeypatch.setenv('ZENODO_TOKEN', 'SECRET')
    monkeypatch.setattr(uz.urllib.request, 'urlopen', fake)
    assert uz.main([str(man), '--drafts-only']) == 0
    st = json.loads(Path(state_path).read_text())['one']
    assert st['reserved_doi'].startswith('10.5281/zenodo.')
    dep = fake.deps[st['id']]
    assert dep['files'] == {} and not dep['submitted'] and dep['metadata']['title'] == 'T'
    assert not [c for c in fake.calls if c[1].startswith('/files/')]
    assert 'nothing published' in capsys.readouterr().out
    assert uz.main([str(man)]) == 0                                     # a normal run then uploads into the same draft
    assert set(fake.deps[st['id']]['files']) == {'a.zip', 'b.zip'}
    assert len(fake.deps) == 1


def _two_records(tmp_path):
    recs = []
    for n in ('one', 'two'):
        f = tmp_path / f'{n}.zip'
        f.write_bytes(n.encode() * 300)
        recs.append({'name': n, 'metadata': dict(META, title=f'T {n}'), 'files': [str(f)]})
    man = tmp_path / 'm.json'
    man.write_text(json.dumps({'records': recs}))
    return recs, man


def test_runs_leave_published_records_alone(tmp_path, monkeypatch, capsys):
    recs, man = _two_records(tmp_path)
    fake = FakeZenodo()
    monkeypatch.setenv('ZENODO_TOKEN', 'SECRET')
    monkeypatch.setattr(uz.urllib.request, 'urlopen', fake)
    assert uz.main([str(man), '--drafts-only']) == 0
    state = json.loads(Path(str(man) + '.state.json').read_text())
    uz.Zenodo('SECRET', opener=fake).publish(state['one']['id'])                     # the owner published one by hand
    fake.calls.clear()
    assert uz.main([str(man)]) == 0                                                  # the full run must not trip on it
    out = capsys.readouterr().out
    assert 'one: already published, left alone' in out
    assert not [c for c in fake.calls if c[0] == 'PUT' and f"/{state['one']['id']}" in c[1]]
    assert set(fake.deps[state['two']['id']]['files']) == {'two.zip'}
    assert uz.main([str(man), '--publish']) == 0                                     # only the other one is published now
    assert 'one: already published' in capsys.readouterr().out
    assert fake.deps[state['two']['id']]['submitted']


def test_refresh_metadata_updates_a_published_record_and_keeps_its_files(tmp_path, monkeypatch):
    recs, man = _two_records(tmp_path)
    fake = FakeZenodo()
    monkeypatch.setenv('ZENODO_TOKEN', 'SECRET')
    monkeypatch.setattr(uz.urllib.request, 'urlopen', fake)
    assert uz.main([str(man)]) == 0
    state = json.loads(Path(str(man) + '.state.json').read_text())
    uz.Zenodo('SECRET', opener=fake).publish(state['one']['id'])
    files_before = dict(fake.deps[state['one']['id']]['files'])
    recs[0]['metadata']['title'] = 'T one, better keywords'
    recs[1]['metadata']['title'] = 'T two, better keywords'
    man.write_text(json.dumps({'records': recs}))
    assert uz.main([str(man), '--refresh-metadata']) == 0
    one, two = fake.deps[state['one']['id']], fake.deps[state['two']['id']]
    assert one['metadata']['title'].endswith('better keywords') and one['submitted'] and not one['editing']
    assert one['files'] == files_before
    assert two['metadata']['title'].endswith('better keywords') and not two['submitted']


def test_refresh_refuses_to_republish_if_files_changed(tmp_path):
    recs, man = _two_records(tmp_path)
    fake = FakeZenodo()
    api = uz.Zenodo('SECRET', opener=fake)
    state = {}
    uz.upload_record(api, recs[0], state, str(man) + '.state.json', log=lambda *_: None)
    api.publish(state['one']['id'])
    real_edit = api.edit

    def edit_and_change(dep_id):
        res = real_edit(dep_id)
        fake.deps[dep_id]['files']['one.zip']['checksum'] = 'changed'
        return res
    api.edit = edit_and_change
    with pytest.raises(RuntimeError, match='files differ'):
        uz.refresh_metadata(api, recs[0], state['one'], log=lambda *_: None)
