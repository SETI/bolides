# CLAUDE.md

Guidance for working in the `bolides` repository.

## What this package is

`bolides` is a public, community Python package for loading **bolide** (bright
fireball/meteor) detections from many sources into a single common format and
analyzing them. Its primary data source is the GLM (Geostationary Lightning
Mapper) bolide detections published at
[neo-bolide.ndc.nasa.gov](https://neo-bolide.ndc.nasa.gov), but it also reads
US Government sensor data, Global Meteor Network data, GLM detection-pipeline
ZODB databases, and user CSVs.

It is published on PyPI (`pip install bolides`) and documented at
[bolides.readthedocs.io](https://bolides.readthedocs.io). Development is
supported by NASA's Asteroid Threat Assessment Project (ATAP).

- **Canonical upstream:** `github.com/SETI/bolides`. Open PRs and issues here.
  (Note: `pyproject.toml` currently lists `jcsmithhere/bolides` URLs and the
  README has a few `jcsmithhere` links — these are stale; SETI is canonical.)

## Core architecture

The whole package is built around two DataFrame subclasses. Understand these
first.

- **`BolideDataFrame`** (`bolides/bdf.py`) — the central class, a subclass of
  GeoPandas `GeoDataFrame`. All bolide data, regardless of source, is loaded
  into this common format (always EPSG:4326 / WGS84, with a `geometry` column
  of points and a `datetime` column in UTC). Construction dispatches on a
  `source=` kwarg in `__init__`. Key methods: `annotate`, `filter_date`,
  `filter_boundary`, `filter_observation`, `filter_shower`, `get_closest`
  (and `_by_time` / `_by_loc`), `augment`, `add_website_data`, and the
  `plot_*` family.
- **`ShowerDataFrame`** (`bolides/sdf.py`) — subclass of pandas `DataFrame`
  for meteor shower data from the IAU Meteor Data Center. Used by
  `BolideDataFrame.filter_shower` and `plot_dates`.

Both classes override `_constructor`, `__getitem__`, and `__setattr__` so that
pandas/geopandas operations preserve the subclass instead of silently
downcasting. When a geopandas method drops back to a base class, the code calls
the module-level `force_bdf_class()` / `force_showers_class()` helpers to
re-stamp `__class__`. **Preserve this pattern** when adding methods that return
filtered/derived frames.

### Supporting modules

- `bolides/sources.py` — the loaders behind each `source=` value:
  `glm`/`website` (neo-bolide API), `usg` (JPL CNEOS fireball API), `gmn`
  (Global Meteor Network), `csv`, `pickle`, `remote` (CSV by URL), and
  `pipeline`/`glm-pipeline` (ZODB, via `pipeline_utils.py`). Every loader
  returns a `GeoDataFrame` via `add_geometry()`. `glm_website_event()` pulls
  per-event light curves and integrated energies.
- `bolides/fov_utils.py` — fields of view. `get_boundary()` returns shapely
  polygons (in an Azimuthal Equidistant CRS, central lat 90) for named sensor
  FOVs: `goes`, `goes-e`, `goes-w`, `goes-w-i`/`-ni` (GOES-17 yaw-flip
  orientations), `goes-17-89.5`, `fy4a*`, `gmn-*km`. This is the source of
  truth for what each satellite can see.
- `bolides/astro_utils.py` — astronomy helpers: lunar phase, solar
  time/altitude (via `ephem`), radiant from velocity, orbital elements (shells
  out to WesternMeteorPyLib), solar-longitude↔date, `haversine`, and
  `_distance_metric` (the space-time improbability metric used by
  `get_closest`).
- `bolides/plotting.py` — matplotlib + cartopy scatter/density plots and the
  plotly interactive plot. `bolides/crs.py` — cartopy projection subclasses
  (`DefaultCRS`, `GOES_E`, `GOES_W`, `FY4A`). `bolides/constants.py` —
  satellite sub-longitudes.
- `bolides/bolide.py`, `bolides/bolidelist.py` — the **original/legacy** API
  (`Bolide`, `BolideList`, `AMSBolideList`) predating `BolideDataFrame`. Still
  shipped and functional, but new work generally goes through
  `BolideDataFrame`.

### Data and metadata files

- `bolides/data/` — packaged data: `GLM_FOV_edges.nc` (FOV vertices),
  `glm16_obs.csv` / `glm17_obs.csv` / `glm18_obs.csv` / `glm19_obs.csv`
  (when each satellite was active and in which FOV orientation), GMN
  shapefiles. The `glm*_obs.csv` files have a leading `#` comment line, so
  they are read with `pd.read_csv(..., header=1)`.
- `bolides/metadata/` — `columns.csv` (human-readable column descriptions
  surfaced by `BolideDataFrame.describe()`) and `*.html` attribution blurbs
  shown in `_repr_html_`.

`bolides/version.py` and `pyproject.toml` both carry the version — keep them in
sync (a past commit fixed them disagreeing).

### Other directories

- `webapp/` — a Dash/Flask interactive map app (deployed at bolides.seti.org).
  Optional `webapp` extra. See `webapp/README.txt`.
- `docs/` — Sphinx documentation (numpydoc style), built for ReadTheDocs.
  `notebooks/` and `binder/` — the tutorial notebook and its Binder config.

## Conventions

Match the surrounding code rather than imposing new tooling.

- **Docstrings:** numpydoc / NumPy-style (Parameters / Returns / Other
  Parameters sections), with cross-references like `` `~BolideDataFrame` ``.
  Public methods are documented this way and the docstrings feed the Sphinx
  reference pages — keep them accurate when you change signatures.
- **Comments:** the codebase is heavily and plainly commented, explaining the
  *why* of each block. Lines are kept to roughly 100 characters.
- **Heavy imports are local:** cartopy, plotly, netCDF4, poliastro, pyproj,
  ZODB, cv2, etc. are imported inside the functions/methods that use them, not
  at module top level, to keep `import bolides` fast and optional deps optional.
  Follow this when adding features that pull in heavy or optional dependencies.
- **Input handling:** methods normalize inputs (e.g. wrap a lone string in a
  list), validate against an explicit list of valid values with a helpful
  `ValueError`, and use `reconcile_input(user_kwargs, defaults)` (in
  `utils.py`) to apply defaults without clobbering user values.
- Times are UTC throughout; naive datetimes are assumed UTC and localized.

## Testing & verification

**There is currently no automated test suite, no CI, and no linter config.**
Verify changes manually — e.g. load a `BolideDataFrame` from a source, exercise
the method you changed, and/or run the tutorial notebooks in `notebooks/`
(`tutorial_BolideDataFrame.ipynb` for the `BolideDataFrame` API,
`tutorial_FOV_tools.ipynb` for the standalone `fov_utils` helpers).

**Validate primarily against GLM data**, the package's main dataset, loaded live
from the NASA neo-bolide website:

```python
bdf = BolideDataFrame(source='glm')
```

It downloads quickly. Prefer this over other sources so tests exercise the
real data path users rely on. (Watch for upstream API format changes — e.g. the
neo-bolide `datetime` field switched from ISO-8601 strings to Unix epoch
milliseconds, which the `glm_website` loader in `sources.py` now handles.) Most
loaders hit live network APIs (neo-bolide, JPL, GMN, IAU); for offline-only
checks, `source='csv'` or `'pickle'` work without network.

> Adding a pytest suite + GitHub Actions CI + pre-commit is tracked in
> **issue #28**. When that lands, update this section with the real commands.

## Dependencies / install notes

Core deps are in `pyproject.toml`. Cartopy/Shapely need system libraries
(PROJ, GEOS) and are finicky via pip — conda-forge cartopy is the smoothest
path (see `README.rst` "Installation"). Optional extras: `pipeline`
(ZODB, zc.zlibstorage) and `webapp` (dash, flask_caching, gunicorn). Install
for development with `pip install -e .`.

## Housekeeping note

The working tree contains editor/backup cruft (`*.swp`, `*~` files, a
`.gitignore~`). Don't commit these, and don't treat the `~`-suffixed files as
source.
