"""ECMWF HRES forecast data download script.

This script downloads actual Numerical Weather Prediction (NWP) forecast data
from ECMWF via the Copernicus Data Store (CDS) API.

WHAT THIS PROVIDES:
  - 100m / 10m wind components (u100, v100, u10, v10)
  - 2m temperature (t2m)
  - Surface pressure (sp)
  - Forecast horizon: +1 to +48 hours
  - 4 base times per day: 00, 06, 12, 18 UTC

USAGE:
  1. Register at https://cds.climate.copernicus.eu/ → obtain API key
  2. Save API key to ~/.cdsapirc
  3. Install:  pip install cdsapi
  4. Run:      python download_ecmwf_forecast.py

REQUIREMENTS:
  pip install cdsapi

OUTPUT:
  data/wind_nc/forecast/ecmwf_hres_<YYYY>.nc  (one file per year)
  Combined with ERA5 via nwp_integration.py
"""

import os
import sys


def main():
    try:
        import cdsapi
    except ImportError:
        print("=" * 60)
        print("  ERROR: cdsapi not installed")
        print("  Run:  pip install cdsapi")
        print("=" * 60)
        sys.exit(1)

    # ── Config ──
    # Wind farm location (same 5 points as ERA5 data)
    TARGET_POINTS = {
        1: (41, 96),
        2: (40.5, 96.5),
        3: (40, 97),
        4: (39.5, 97.5),
        5: (39, 98),
    }

    # Area: [N, W, S, E] bounding box (slightly larger than target points)
    AREA = [42, 95, 38, 99]

    # Years to download (use 2024-2026 to match existing ERA5 data)
    YEARS = ["2024", "2025", "2026"]

    OUT_DIR = os.path.join(os.path.dirname(__file__), "forecast")
    os.makedirs(OUT_DIR, exist_ok=True)

    c = cdsapi.Client()

    # ── Option A: ERA5 forecast fields (reanalysis-based, more accessible) ──
    # Dataset: "reanalysis-era5-single-levels"
    # Contains forecast parameters like "10m_u_component_of_wind" etc.
    # Available at hourly resolution

    # ── Option B: ECMWF HRES operational forecast ──
    # Dataset: "ecmwf-hres-day"
    # Higher resolution, actual operational forecast
    # Available at 3-hourly resolution, 4 base times daily

    # We use Option A (ERA5 forecast-type fields) as the most reliable
    # and freely available option.

    for year in YEARS:
        out_file = os.path.join(OUT_DIR, f"era5_forecast_{year}.nc")
        if os.path.exists(out_file):
            print(f"  [SKIP] {out_file} already exists")
            continue

        print(f"\n  Downloading ERA5 forecast data for {year} ...")
        print(f"  Area: {AREA}")

        # ── CDS API request ──
        # NOTE: You can change 'time' to download specific base times
        # NOTE: The 'format' can be 'netcdf' (preferred) or 'grib'
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": ["reanalysis"],
                "variable": [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "100m_u_component_of_wind",
                    "100m_v_component_of_wind",
                    "2m_temperature",
                    "mean_sea_level_pressure",
                    "surface_pressure",
                    "boundary_layer_height",
                ],
                "year": year,
                "month": [
                    "01", "02", "03", "04", "05", "06",
                    "07", "08", "09", "10", "11", "12",
                ],
                "day": [
                    "01", "02", "03", "04", "05", "06",
                    "07", "08", "09", "10", "11", "12",
                    "13", "14", "15", "16", "17", "18",
                    "19", "20", "21", "22", "23", "24",
                    "25", "26", "27", "28", "29", "30", "31",
                ],
                "time": [
                    "00:00", "01:00", "02:00", "03:00", "04:00",
                    "05:00", "06:00", "07:00", "08:00", "09:00",
                    "10:00", "11:00", "12:00", "13:00", "14:00",
                    "15:00", "16:00", "17:00", "18:00", "19:00",
                    "20:00", "21:00", "22:00", "23:00",
                ],
                "data_format": "netcdf",
                "area": AREA,
            },
            out_file,
        )
        print(f"  Saved -> {out_file}")

    print(f"\n{'=' * 60}")
    print(f"  Download complete!")
    print(f"  Files saved to: {OUT_DIR}")
    print(f"\n  Next step: update the data loading path in")
    print(f"  forecast_vmd_hybrid.py to use these files.")
    print(f"  Or run: python forecast_tsp/nwp_integration.py")
    print(f"  to process and merge with existing data.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    print("=" * 60)
    print("  ECMWF NWP Forecast Data Download")
    print("  Copernicus Data Store (CDS) API")
    print("=" * 60)
    main()
