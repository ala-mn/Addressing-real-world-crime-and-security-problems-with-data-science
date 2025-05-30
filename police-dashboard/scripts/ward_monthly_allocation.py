#!/usr/bin/env python3
import os
import pandas as pd

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PRED_CSV = os.path.join(DATA_DIR, "last3months.csv")  # stub for next-3-mo preds
OUT_CSV  = "ward_monthly_allocation.csv"

# Hours per ward per month capacity:
WEEKLY_HOURS = 800    # 100 officers × 2 hrs/day × 4 days
MONTHLY_HOURS = WEEKLY_HOURS * 4

# ─── LOAD & AGGREGATE TO WARD × MONTH ─────────────────────────────────────────
# last3months.csv has lsoa_code, lsoa_name, year_month, burglary_count
lsoa = pd.read_csv(PRED_CSV)
# you’ll need a mapping LSOA → ward; for now assume your CSV has ward_code already,
# otherwise join on your ward_lookup table here.
# If using LSOA, you’d spatial-join as before; here we assume ward_code exists:
# lsoa['ward_code'] = …  
# For simplicity, let’s pretend last3months.csv is already ward-level:
ward_month = (lsoa
  .groupby(['ward_code','year_month'])
  .burglary_count
  .sum()
  .reset_index(name='predicted_count')
)

# ─── PROPORTIONAL MONTHLY ALLOCATION ───────────────────────────────────────────
rows = []
for ward, grp in ward_month.groupby('ward_code'):
    total = grp['predicted_count'].sum()
    if total == 0:
        # evenly split if no predicted burglaries
        share = MONTHLY_HOURS / len(grp)
        for _,r in grp.iterrows():
            rows.append({
                'ward_code': ward,
                'year_month': r['year_month'],
                'allocated_hours': share
            })
    else:
        for _,r in grp.iterrows():
            alloc = MONTHLY_HOURS * (r['predicted_count'] / total)
            rows.append({
                'ward_code': ward,
                'year_month': r['year_month'],
                'allocated_hours': alloc
            })

alloc_df = pd.DataFrame(rows)

# ─── OUTPUT ──────────────────────────────────────────────────────────────────
alloc_df.to_csv(OUT_CSV, index=False)
print(f"Wrote {OUT_CSV} ({len(alloc_df)} rows: #wards×#months)")
