# Code signing policy

Free code signing provided by [SignPath.io](https://signpath.io/), certificate by the
[SignPath Foundation](https://signpath.org/).

Signed artifacts: the Windows `OriginStack.exe` (in the packaged zip) and the
`OriginStack-*-setup.exe` installer, both produced by
[`packaging/build_windows.ps1`](packaging/build_windows.ps1) and attached to
[GitHub releases](https://github.com/hd152/originstack/releases). Signing runs via
[`.github/workflows/release.yml`](.github/workflows/release.yml)'s `signpath/github-action-submit-signing-request`
step, submitted straight from that CI build with no manual repackaging. Bundled `.pyd`/`.dll`
files inside the onedir folder stay unsigned. See
[`packaging/README.md`](packaging/README.md#code-signing-signpath-foundation) for the full
setup.

## Privacy

OriginStack is a local command-line/desktop pipeline. It does not run as a server and does not
collect telemetry. It makes outbound network requests **only** in the specific cases below, each
of which can be turned off entirely with `--offline` (a checkbox in the desktop app):

| What | When | What is sent |
|------|------|--------------|
| SIMBAD lookup of the target | Normal run, when the object name isn't in the built-in table | The name string only |
| astrometry.net | `--plate-solve` (astrometry.net mode) | The stacked image, to solve its position (needs your own API key) |
| Gaia / VizieR / SIMBAD catalogues | `--photometry`, `--photometry-timeseries`, `--annotate`, colour calibration | Sky coordinates of the field only |
| JPL Horizons | Comet ephemerides | The comet designation and time only |
| GitHub releases API | Once per CLI run or desktop-app launch (self-update check) | Nothing — anonymous GET, no parameters, no identifying data |

No frame data is ever uploaded except the astrometry.net case above, which is opt-in per run via
an explicit flag. Full detail, including how to disable each case individually, is in the
README's [Network use and `--offline`](README.md#network-use-and---offline) section and in
[`SECURITY.md`](SECURITY.md).

Third-party services this project can talk to, and their own privacy policies:
[astrometry.net](https://astrometry.net) (no published policy; treat uploaded images as sent to
a third party), [CDS](https://cds.unistra.fr) (SIMBAD/VizieR — no formal privacy policy
published; queries carry only coordinates, names or designations, never image data),
[Gaia / ESA](https://www.cosmos.esa.int/web/gaia/privacy) (via the CDS/VizieR mirror above),
[JPL/NASA](https://www.jpl.nasa.gov/jpl-privacy-policy), and
[GitHub](https://docs.github.com/en/site-policy/privacy-policies/github-privacy-statement) (the
self-update check's public releases API).

## Roles

OriginStack is solo-maintained. [Tom / Hans Davenport](https://github.com/hd152) (one GitHub
account, `hd152`) holds all three SignPath roles:

- **Author** — commits directly without requiring review.
- **Reviewer** — approves any change from a non-author contributor before merge.
- **Approver** — the only person authorized to approve SignPath signing requests.

There is no separate reviewer/approver team at this time; this will be updated here if that
changes.
