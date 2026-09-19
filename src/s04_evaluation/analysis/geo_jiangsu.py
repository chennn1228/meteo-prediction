# -*- coding: utf-8 -*-
"""Jiangsu geography helper: load boundary, draw outline, mask grid, site regions.

Boundary source: DataV GeoJSON (data/00_geo/jiangsu.geojson), 13 city polygons.
Province outline = unary_union of cities.
"""
import json
from pathlib import Path
import numpy as np
import yaml
from shapely.geometry import shape
from shapely.ops import unary_union
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

CODE_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / "config" / "01_sites.yaml").exists())
GEOJSON = CODE_ROOT / "data" / "00_geo" / "jiangsu.geojson"
SITES_YAML = CODE_ROOT / "config" / "01_sites.yaml"

_cache = {}

def load_jiangsu():
    """Return (province_geom, list_of_city_geoms, list_of_city_names)."""
    if "js" in _cache:
        return _cache["js"]
    g = json.loads(GEOJSON.read_text(encoding="utf-8"))
    cities = [shape(f["geometry"]) for f in g["features"]]
    names = [f["properties"].get("name", "") for f in g["features"]]
    prov = unary_union(cities)
    _cache["js"] = (prov, cities, names)
    return _cache["js"]

def _rings(geom):
    """Yield exterior+interior coordinate arrays for Polygon/MultiPolygon."""
    polys = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
    for p in polys:
        yield np.array(p.exterior.coords)
        for r in p.interiors:
            yield np.array(r.coords)

def add_jiangsu_outline(ax, show_cities=True, lw_prov=1.1, lw_city=0.4,
                        color="#333333", city_color="#999999"):
    """Draw province outer boundary (thick) and optional city borders (thin)."""
    prov, cities, _ = load_jiangsu()
    if show_cities:
        for c in cities:
            for ring in _rings(c):
                ax.plot(ring[:, 0], ring[:, 1], lw=lw_city, color=city_color, zorder=2)
    for ring in _rings(prov):
        ax.plot(ring[:, 0], ring[:, 1], lw=lw_prov, color=color, zorder=3)
    ax.set_aspect("equal", adjustable="box")

def mask_to_jiangsu(lon2d, lat2d):
    """Boolean grid mask: True where inside province."""
    prov, _, _ = load_jiangsu()
    x = lon2d.ravel(); y = lat2d.ravel()
    try:
        import shapely
        inside = shapely.contains_xy(prov, x, y)
    except Exception:
        from shapely.geometry import Point
        inside = np.array([prov.contains(Point(px, py)) for px, py in zip(x, y)])
    return np.asarray(inside).reshape(lon2d.shape)

def contains_point(lon, lat):
    prov, _, _ = load_jiangsu()
    from shapely.geometry import Point
    return bool(prov.contains(Point(lon, lat)))

def site_regions():
    """Return dict station_id -> region name from config/01_sites.yaml."""
    cfg = yaml.safe_load(SITES_YAML.read_text(encoding="utf-8"))
    out = {}
    for s in cfg.get("sites", []):
        if isinstance(s, dict) and "id" in s:
            out[s["id"]] = s.get("region") or s.get("layer") or s.get("code") or "unknown"
    return out

def site_codes():
    """Return dict station_id -> ASCII layer code (e.g. SI/MI/MC/NI/NC)."""
    cfg = yaml.safe_load(SITES_YAML.read_text(encoding="utf-8"))
    return {s["id"]: s.get("code", s.get("region", "?"))
            for s in cfg.get("sites", []) if isinstance(s, dict) and "id" in s}
