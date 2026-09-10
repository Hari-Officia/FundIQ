import sys
import time
import gc
import numpy as np
import polars as pl
import xgboost as xgb

sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 60)
    print("ACCURACY BOOST EXPERIMENTAL PIPELINE")
    print("=" * 60)
    
    t0 = time.time()
    df = pl.read_parquet(r'ML data/nav_history_2020_2025_raw.parquet', columns=['scheme_code', 'scheme_name', 'date', 'nav'])
    df = df.with_columns([
        pl.col('date').str.to_date(),
        pl.col('nav').cast(pl.Float32)
    ]).sort(['scheme_code', 'date'])

    df = df.with_columns(pl.when(pl.col('nav') > 0).then(pl.col('nav')).otherwise(None).alias('valid_nav'))

    # Base Returns
    df = df.with_columns([
        (pl.col('valid_nav') / pl.col('valid_nav').shift(1).over('scheme_code') - 1).cast(pl.Float32).alias('ret_1d'),
        (pl.col('valid_nav') / pl.col('valid_nav').shift(5).over('scheme_code') - 1).cast(pl.Float32).alias('ret_5d'),
        (pl.col('valid_nav') / pl.col('valid_nav').shift(20).over('scheme_code') - 1).cast(pl.Float32).alias('ret_20d'),
        (pl.col('valid_nav') / pl.col('valid_nav').shift(60).over('scheme_code') - 1).cast(pl.Float32).alias('ret_60d'),
        (pl.col('valid_nav').shift(-5).over('scheme_code') / pl.col('valid_nav') - 1).cast(pl.Float32).alias('target_5obs')
    ])

    # Category Detection (Equity, Debt, Hybrid, Other)
    print("Detecting scheme categories...")
    name_col = pl.col('scheme_name').str.to_lowercase()
    is_equity = (name_col.str.contains('equity|small cap|mid cap|large cap|flexi cap|multi cap|index|elss'))
    is_debt = (name_col.str.contains('debt|bond|liquid|money market|gilt|overnight|credit risk|corporate bond'))
    is_hybrid = (name_col.str.contains('hybrid|balanced'))
    
    df = df.with_columns([
        pl.when(is_equity).then(1)
          .when(is_debt).then(2)
          .when(is_hybrid).then(3)
          .otherwise(0).cast(pl.Int8).alias('category_id')
    ])

    # Date-level market relative returns
    print("Computing market-relative returns...")
    df = df.with_columns([
        (pl.col('ret_5d') - pl.col('ret_5d').mean().over('date')).cast(pl.Float32).alias('ret_5d_rel'),
        (pl.col('ret_20d') - pl.col('ret_20d').mean().over('date')).cast(pl.Float32).alias('ret_20d_rel')
    ])

    # Rolling indicators
    df = df.with_columns([
        pl.col('ret_1d').rolling_std(20, min_samples=20).over('scheme_code').cast(pl.Float32).alias('vol_20d'),
        pl.col('ret_1d').rolling_std(60, min_samples=60).over('scheme_code').cast(pl.Float32).alias('vol_60d'),
        (pl.col('valid_nav') / pl.col('valid_nav').rolling_mean(20, min_samples=20).over('scheme_code') - 1).cast(pl.Float32).alias('price_vs_ma20'),
        (pl.col('valid_nav') / pl.col('valid_nav').rolling_mean(60, min_samples=60).over('scheme_code') - 1).cast(pl.Float32).alias('price_vs_ma60'),
        (pl.col('valid_nav') / pl.col('valid_nav').rolling_mean(5, min_samples=5).over('scheme_code') - 1).cast(pl.Float32).alias('price_vs_ma5')
    ])

    df = df.with_columns([
        (pl.col('vol_20d') / (pl.col('vol_60d') + 1e-6)).cast(pl.Float32).alias('vol_ratio')
    ])

    feature_cols = [
        'category_id', 'ret_1d', 'ret_5d', 'ret_20d', 'ret_60d',
        'ret_5d_rel', 'ret_20d_rel', 'vol_20d', 'vol_60d',
        'price_vs_ma20', 'price_vs_ma60', 'price_vs_ma5', 'vol_ratio'
    ]

    valid_cond = pl.col('target_5obs').is_not_null() & pl.col('target_5obs').is_finite() & pl.col('target_5obs').is_between(-0.99, 5.0)
    for c in feature_cols:
        valid_cond = valid_cond & pl.col(c).is_not_null() & pl.col(c).is_finite()

    df_clean = df.filter(valid_cond)
    print(f"Clean sample rows: {len(df_clean):,}")

    train_df = df_clean.filter(pl.col('date').dt.year() <= 2023)
    val_df = df_clean.filter(pl.col('date').dt.year() == 2024)
    test_df = df_clean.filter(pl.col('date').dt.year() == 2025)

    X_tr = train_df.select(feature_cols).to_numpy().astype(np.float32)
    y_tr_b = (train_df['target_5obs'].to_numpy() > 0).astype(np.int32)

    X_va = val_df.select(feature_cols).to_numpy().astype(np.float32)
    y_va_b = (val_df['target_5obs'].to_numpy() > 0).astype(np.int32)
    y_va_reg = val_df['target_5obs'].to_numpy().astype(np.float32)

    X_te = test_df.select(feature_cols).to_numpy().astype(np.float32)
    y_te_b = (test_df['target_5obs'].to_numpy() > 0).astype(np.int32)
    y_te_reg = test_df['target_5obs'].to_numpy().astype(np.float32)

    print("\n--- EXPERIMENT 1: XGBClassifier (Direct Binary Directional Model) ---")
    clf = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        n_jobs=4,
        early_stopping_rounds=30,
        random_state=42
    )
    clf.fit(X_tr, y_tr_b, eval_set=[(X_va, y_va_b)], verbose=100)

    val_preds_prob = clf.predict_proba(X_va)[:, 1]
    test_preds_prob = clf.predict_proba(X_te)[:, 1]

    val_acc_full = (clf.predict(X_va) == y_va_b).mean()
    test_acc_full = (clf.predict(X_te) == y_te_b).mean()

    print(f"Validation Full Classifier Accuracy: {val_acc_full*100:.2f}%")
    print(f"Test Full Classifier Accuracy:       {test_acc_full*100:.2f}%")

    # High Confidence Thresholding (Predicting when Probability > 0.55 or < 0.45)
    high_conf_mask_val = (val_preds_prob > 0.55) | (val_preds_prob < 0.45)
    val_acc_hc = (clf.predict(X_va)[high_conf_mask_val] == y_va_b[high_conf_mask_val]).mean()

    high_conf_mask_test = (test_preds_prob > 0.55) | (test_preds_prob < 0.45)
    test_acc_hc = (clf.predict(X_te)[high_conf_mask_test] == y_te_b[high_conf_mask_test]).mean()

    print(f"\n--- HIGH CONFIDENCE MARGIN ACCURACY (>55% or <45%) ---")
    print(f"Validation High Confidence Accuracy: {val_acc_hc*100:.2f}% ({high_conf_mask_val.sum():,} samples)")
    print(f"Test High Confidence Accuracy:       {test_acc_hc*100:.2f}% ({high_conf_mask_test.sum():,} samples)")

    # Category Breakdown for Debt Funds
    debt_mask_test = (X_te[:, 0] == 2)
    debt_acc_test = (clf.predict(X_te)[debt_mask_test] == y_te_b[debt_mask_test]).mean()
    print(f"\nDebt Fund Category Accuracy (Test): {debt_acc_test*100:.2f}% ({debt_mask_test.sum():,} samples)")

    equity_mask_test = (X_te[:, 0] == 1)
    equity_acc_test = (clf.predict(X_te)[equity_mask_test] == y_te_b[equity_mask_test]).mean()
    print(f"Equity Fund Category Accuracy (Test): {equity_acc_test*100:.2f}% ({equity_mask_test.sum():,} samples)")

if __name__ == "__main__":
    main()
