AI-SDN offline demo (open demo/index.html in any browser, no install)

1. Refresh the numbers (and load real Abilene traffic) from the repo root:
     .venv/bin/python scripts/export_demo_data.py
   This rewrites demo/data.js from results/tables and data/raw/abilene.
   Without the raw files the demo still works and uses a synthetic spike.
2. Open demo/index.html.

What is what
- M1, M2, M3 pages: simple in-browser simulation (no trained models, no controller).
- M3 page and Results page: measured numbers from results/tables (via data.js).
