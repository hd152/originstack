#!/usr/bin/env python3
"""Upload data sets to Zenodo as private drafts, and publish them only when told to.

    python tools/upload_zenodo.py MANIFEST.json                 # create drafts and upload the files
    python tools/upload_zenodo.py MANIFEST.json --publish       # publish every draft of the manifest
    python tools/upload_zenodo.py MANIFEST.json --publish NAME  # publish one
    python tools/upload_zenodo.py MANIFEST.json --status        # what each record's draft looks like
    python tools/upload_zenodo.py MANIFEST.json --drafts-only   # create the drafts (reserved DOIs), no files
    python tools/upload_zenodo.py MANIFEST.json --refresh-metadata  # new metadata, also on published records

The manifest lists the records::

    {"records": [{"name": "01_galaxy", "metadata": {...Zenodo deposit metadata...},
                  "files": ["G:/zips/01_galaxy.zip", ...]}]}

``metadata`` is Zenodo's own deposit metadata (title, upload_type, description, creators, license,
keywords, ...). Nothing in it is invented here: what you write is what is sent.

The token is read from the ``ZENODO_TOKEN`` environment variable, or on Windows from the user
environment (``setx ZENODO_TOKEN ...`` needs no restart). ``--sandbox`` uses sandbox.zenodo.org and
``ZENODO_SANDBOX_TOKEN``. The token is never printed or written anywhere.

Drafts are private and can be deleted on the Zenodo site; a *published* record cannot be deleted
and its files are fixed (changes mean a new version), so publishing is a separate, explicit step
that first prints what it is about to do and refuses a record whose uploaded files do not match the
local ones (size and MD5, which Zenodo computes).

State (draft ids and which files are uploaded) is kept in ``MANIFEST.json.state.json`` so an
interrupted run resumes: files already uploaded with the right size and checksum are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional

API = {'live': 'https://zenodo.org/api', 'sandbox': 'https://sandbox.zenodo.org/api'}
TOKEN_VAR = {'live': 'ZENODO_TOKEN', 'sandbox': 'ZENODO_SANDBOX_TOKEN'}
RETRIES = 4


def get_token(env: str = 'live') -> Optional[str]:
    """The token for ``env`` from the process environment, else the Windows user environment."""
    name = TOKEN_VAR[env]
    t = os.environ.get(name)
    if t:
        return t.strip()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as k:
            return str(winreg.QueryValueEx(k, name)[0]).strip()
    except (ImportError, OSError):
        return None


class Zenodo:
    def __init__(self, token: str, env: str = 'live', opener=None):
        self.base = API[env]
        self._token = token
        self._open = opener or urllib.request.urlopen

    def _call(self, method: str, url: str, body: Optional[dict] = None, data=None,
              headers: Optional[dict] = None, length: Optional[int] = None):
        h = {'Authorization': 'Bearer ' + self._token}
        payload = data
        if body is not None:
            payload = json.dumps(body).encode()
            h['Content-Type'] = 'application/json'
        if headers:
            h.update(headers)
        if length is not None:
            h['Content-Length'] = str(length)
        last = None
        for attempt in range(RETRIES):
            req = urllib.request.Request(url, data=payload, headers=h, method=method)
            try:
                with self._open(req, timeout=600) as r:
                    raw = r.read()
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as e:
                detail = e.read()[:400].decode('utf-8', 'replace')
                if e.code in (429, 500, 502, 503, 504) and attempt < RETRIES - 1:
                    time.sleep(2 ** attempt * 3)
                    last = f'HTTP {e.code}'
                    if hasattr(payload, 'seek'):
                        payload.seek(0)
                    continue
                raise RuntimeError(f'Zenodo {method} {url.split("/api")[-1]} -> HTTP {e.code}: {detail}') from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = str(e)
                if attempt < RETRIES - 1:
                    time.sleep(2 ** attempt * 3)
                    if hasattr(payload, 'seek'):
                        payload.seek(0)
                    continue
        raise RuntimeError(f'Zenodo {method} failed after {RETRIES} tries: {last}')

    def create(self) -> dict:
        return self._call('POST', f'{self.base}/deposit/depositions', body={})

    def get(self, dep_id: int) -> dict:
        return self._call('GET', f'{self.base}/deposit/depositions/{dep_id}')

    def set_metadata(self, dep_id: int, metadata: dict) -> dict:
        return self._call('PUT', f'{self.base}/deposit/depositions/{dep_id}', body={'metadata': metadata})

    def files(self, dep_id: int) -> List[dict]:
        return self._call('GET', f'{self.base}/deposit/depositions/{dep_id}/files')

    def upload(self, bucket_url: str, path: str) -> dict:
        name = os.path.basename(path)
        size = os.path.getsize(path)
        with open(path, 'rb') as fh:
            return self._call('PUT', f'{bucket_url}/{urllib.parse.quote(name)}', data=fh, length=size,
                              headers={'Content-Type': 'application/octet-stream'})

    def publish(self, dep_id: int) -> dict:
        return self._call('POST', f'{self.base}/deposit/depositions/{dep_id}/actions/publish')

    def edit(self, dep_id: int) -> dict:
        """Reopen a published record for a metadata change (its files stay fixed)."""
        return self._call('POST', f'{self.base}/deposit/depositions/{dep_id}/actions/edit')


def md5_of(path: str, block: int = 1 << 22) -> str:
    h = hashlib.md5()  # noqa: S324 -- Zenodo reports MD5, used only to compare with what it stored
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


def load_state(path: str) -> dict:
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_state(path: str, state: dict) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, path)


def check_manifest(m: dict) -> List[str]:
    """Problems in the manifest that would only show up half-way through an upload."""
    out: List[str] = []
    names = set()
    for r in m.get('records', []):
        n = r.get('name')
        if not n or n in names:
            out.append(f'record name missing or repeated: {n!r}')
        names.add(n)
        md = r.get('metadata') or {}
        for k in ('title', 'upload_type', 'description', 'creators', 'license'):
            if not md.get(k):
                out.append(f'{n}: metadata.{k} is missing')
        for f in r.get('files', []):
            if not os.path.isfile(f):
                out.append(f'{n}: file not found: {f}')
        if not r.get('files'):
            out.append(f'{n}: no files')
    return out


def upload_record(api: Zenodo, rec: dict, state: dict, state_path: str, log=print,
                  files: bool = True) -> int:
    """Create (or reuse) the draft of ``rec``, set its metadata, upload what is not there yet.

    ``files=False`` stops after the metadata: the draft then exists (with its reserved DOI) but holds
    no files yet."""
    name = rec['name']
    st = state.setdefault(name, {})
    if 'id' not in st:
        dep = api.create()
        st.update(id=dep['id'], bucket=dep['links']['bucket'], html=dep['links'].get('html'))
        save_state(state_path, state)
        log(f'  {name}: draft {st["id"]} created')
    dep_id = st['id']
    dep = api.get(dep_id)
    if dep.get('submitted'):
        raise RuntimeError(f'{name}: record {dep_id} is already published; not touching it')
    st['bucket'] = dep['links']['bucket']
    updated = api.set_metadata(dep_id, rec['metadata'])
    doi = ((updated.get('metadata') or {}).get('prereserve_doi') or {}).get('doi')
    if doi:
        st['reserved_doi'] = doi
    save_state(state_path, state)
    if not files:
        return dep_id
    present = {f['filename']: f for f in api.files(dep_id)}
    for path in rec['files']:
        base = os.path.basename(path)
        size = os.path.getsize(path)
        have = present.get(base)
        if have and have.get('filesize') == size and have.get('checksum') == md5_of(path):
            log(f'  {name}: {base} already uploaded ({size / 1e6:.0f} MB)')
            continue
        t = time.time()
        log(f'  {name}: uploading {base} ({size / 1e6:.0f} MB) ...')
        api.upload(st['bucket'], path)
        log(f'  {name}: {base} done in {time.time() - t:.0f} s')
    st['uploaded'] = True
    save_state(state_path, state)
    return dep_id


def published_names(api: Zenodo, state: dict) -> set:
    """Names of the records in ``state`` that are already published."""
    return {name for name, st in state.items() if st.get('id') and api.get(st['id']).get('submitted')}


def refresh_metadata(api: Zenodo, rec: dict, st: dict, log=print) -> None:
    """Send the manifest's metadata to a record. A published record is reopened (``edit``), updated and
    published again: same DOI and files, new metadata. Refuses if the files changed meanwhile."""
    dep_id = st['id']
    dep = api.get(dep_id)
    if dep.get('submitted'):
        before = {f['filename']: f.get('checksum') for f in api.files(dep_id)}
        api.edit(dep_id)
        api.set_metadata(dep_id, rec['metadata'])
        after = {f['filename']: f.get('checksum') for f in api.files(dep_id)}
        if before != after:
            raise RuntimeError(f"{rec['name']}: files differ after reopening; not publishing again")
        api.publish(dep_id)
        log(f"  {rec['name']}: published record {dep_id} updated (metadata only)")
    else:
        api.set_metadata(dep_id, rec['metadata'])
        log(f"  {rec['name']}: draft {dep_id} updated")


def verify_record(api: Zenodo, rec: dict, dep_id: int) -> List[str]:
    """Differences between the local files and what the draft holds (empty = identical)."""
    remote = {f['filename']: f for f in api.files(dep_id)}
    problems = []
    for path in rec['files']:
        base = os.path.basename(path)
        f = remote.get(base)
        if not f:
            problems.append(f'{base}: not on the draft')
        elif f.get('filesize') != os.path.getsize(path):
            problems.append(f'{base}: size {f.get("filesize")} != {os.path.getsize(path)}')
        elif f.get('checksum') != md5_of(path):
            problems.append(f'{base}: checksum differs')
    extra = set(remote) - {os.path.basename(p) for p in rec['files']}
    problems += [f'{b}: on the draft but not in the manifest' for b in sorted(extra)]
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('manifest')
    ap.add_argument('--sandbox', action='store_true', help='use sandbox.zenodo.org and ZENODO_SANDBOX_TOKEN')
    ap.add_argument('--status', action='store_true', help='show the drafts of the manifest and exit')
    ap.add_argument('--publish', nargs='*', metavar='NAME',
                    help='publish the named records (all of them when none are named); IRREVERSIBLE')
    ap.add_argument('--check', action='store_true', help='validate the manifest and exit (no network)')
    ap.add_argument('--refresh-metadata', nargs='*', metavar='NAME',
                    help='send the manifest metadata to the named records (all when none are named), '
                         'including already published ones (reopened, updated, published again; files untouched)')
    ap.add_argument('--drafts-only', action='store_true',
                    help="create each record's draft and set its metadata, but upload no files")
    a = ap.parse_args(argv)
    with open(a.manifest, encoding='utf-8') as fh:
        manifest = json.load(fh)
    problems = check_manifest(manifest)
    if problems:
        sys.stdout.write('Manifest problems:\n' + '\n'.join('  ' + p for p in problems) + '\n')
        return 1
    if a.check:
        sys.stdout.write(f"{len(manifest['records'])} records, "
                         f"{sum(len(r['files']) for r in manifest['records'])} files: manifest ok\n")
        return 0
    env = 'sandbox' if a.sandbox else 'live'
    token = get_token(env)
    if not token:
        sys.exit(f'{TOKEN_VAR[env]} is not set (see the module docstring)')
    api = Zenodo(token, env)
    state_path = a.manifest + '.state.json'
    state = load_state(state_path)
    records = manifest['records']

    if a.status:
        for r in records:
            st = state.get(r['name'])
            if not st:
                print(f"{r['name']}: no draft yet")
                continue
            dep = api.get(st['id'])
            print(f"{r['name']}: record {st['id']} {'PUBLISHED' if dep.get('submitted') else 'draft'}  "
                  f"{dep['links'].get('html', '')}  files={len(api.files(st['id']))}")
        return 0

    if a.publish is not None:
        chosen = [r for r in records if not a.publish or r['name'] in a.publish]
        if not chosen:
            sys.exit('no such record in the manifest')
        done = published_names(api, state)
        for r in chosen:
            if r['name'] in done:
                print(f"  {r['name']}: already published")
        chosen = [r for r in chosen if r['name'] not in done]
        if not chosen:
            print('nothing left to publish')
            return 0
        for r in chosen:
            st = state.get(r['name'])
            if not st:
                sys.exit(f"{r['name']}: no draft to publish")
            bad = verify_record(api, r, st['id'])
            if bad:
                sys.exit(f"{r['name']}: refusing to publish:\n  " + '\n  '.join(bad))
        print('About to publish (this cannot be undone): ' + ', '.join(r['name'] for r in chosen))
        for r in chosen:
            res = api.publish(state[r['name']]['id'])
            state[r['name']]['doi'] = res.get('doi') or res.get('metadata', {}).get('doi')
            state[r['name']]['published'] = True
            save_state(state_path, state)
            print(f"  {r['name']}: published, DOI {state[r['name']]['doi']}")
        return 0

    if a.refresh_metadata is not None:
        chosen = [r for r in records if not a.refresh_metadata or r['name'] in a.refresh_metadata]
        for r in chosen:
            if r['name'] not in state:
                sys.exit(f"{r['name']}: no record yet")
            refresh_metadata(api, r, state[r['name']])
        return 0

    done = published_names(api, state)
    if a.drafts_only:
        for r in records:
            if r['name'] in done:
                continue
            upload_record(api, r, state, state_path, files=False)
            st = state[r['name']]
            print(f"  {r['name']}: draft {st['id']}  DOI {st.get('reserved_doi', '?')}  {st.get('html', '')}")
        print('\nDrafts exist (no files uploaded, nothing published).')
        return 0

    for r in records:
        if r['name'] in done:
            print(f"  {r['name']}: already published, left alone")
            continue
        upload_record(api, r, state, state_path)
        bad = verify_record(api, r, state[r['name']]['id'])
        if bad:
            sys.stdout.write(f"{r['name']}: draft differs from the local files:\n  " + '\n  '.join(bad) + '\n')
            return 1
    print('\nDrafts ready (nothing is published). Review them:')
    for r in records:
        print(f"  {r['name']}: {state[r['name']].get('html') or state[r['name']]['id']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
