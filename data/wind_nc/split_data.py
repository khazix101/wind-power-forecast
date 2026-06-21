import pandas as pd
import os

INPUT_CSV = "data/wind_nc/output/wind_data.csv"
OUTPUT_DIR = "data/wind_nc/output"
POINT_ID = 1

VAL_BLOCKS = [
    ("2025-01-01", "2025-01-25"),
    ("2025-04-01", "2025-04-25"),
    ("2025-07-01", "2025-07-25"),
]

df = pd.read_csv(INPUT_CSV)
df["valid_time"] = pd.to_datetime(df["valid_time"])
df = df[df["point_id"] == POINT_ID].sort_values("valid_time").reset_index(drop=True)

print(f"Total point_id={POINT_ID} samples: {len(df)}")
print(f"Time range: {df['valid_time'].min()}  →  {df['valid_time'].max()}")

# ── Test: 2026 ──
test_mask = df["valid_time"].dt.year == 2026
test_df = df[test_mask].copy()
print(f"Test:  {len(test_df)} rows ({test_df['valid_time'].min()} → {test_df['valid_time'].max()})")

# ── Validation: 3 blocks ──
val_masks = []
for start_str, end_str in VAL_BLOCKS:
    start = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str) + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    mask = (df["valid_time"] >= start) & (df["valid_time"] <= end) & (df["valid_time"].dt.year == 2025)
    val_masks.append(mask)

val_mask_combined = val_masks[0] | val_masks[1] | val_masks[2]
val_df = df[val_mask_combined].copy()
print(f"Val:   {len(val_df)} rows ({val_df['valid_time'].min()} → {val_df['valid_time'].max()})")
for i, (start_str, end_str) in enumerate(VAL_BLOCKS):
    n = val_masks[i].sum()
    print(f"  Block {i+1}: {start_str} ~ {end_str}  ({n} rows)")

# ── Train: 2024 + 2025 minus val blocks ──
train_mask = (df["valid_time"].dt.year.isin([2024, 2025])) & ~val_mask_combined
train_df = df[train_mask].copy()
print(f"Train: {len(train_df)} rows ({train_df['valid_time'].min()} → {train_df['valid_time'].max()})")

# ── Drop unused columns ──
drop_cols = ["latitude", "longitude", "number", "expver", "wind_dir_100m", "t2m_degC", "sp_hPa"]
for col in drop_cols:
    if col in train_df.columns:
        train_df.drop(columns=[col], inplace=True)
        test_df.drop(columns=[col], inplace=True)
        val_df.drop(columns=[col], inplace=True)

# ── Save ──
os.makedirs(OUTPUT_DIR, exist_ok=True)
train_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False, encoding="utf-8-sig")
val_df.to_csv(f"{OUTPUT_DIR}/val.csv", index=False, encoding="utf-8-sig")
test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False, encoding="utf-8-sig")

print(f"\nSaved: {OUTPUT_DIR}/train.csv ({len(train_df)} rows)")
print(f"Saved: {OUTPUT_DIR}/val.csv   ({len(val_df)} rows)")
print(f"Saved: {OUTPUT_DIR}/test.csv  ({len(test_df)} rows)")
