$ErrorActionPreference = "Continue"

# GINN lambda sweep (30 tickers)
foreach ($L in 0.0, 0.1, 0.3, 0.5) {
    Write-Host "lambda = $L"
    python scripts/train_ginn_d3.py --lambda_garch $L --max_tickers 30
}

# RQA config sweep (all 503 stocks)
foreach ($C in "default", "small_window", "large_window", "per_series") {
    Write-Host "config = $C"
    python scripts/train_features_baselines_d3.py --rqa_config $C
}

# Beta-RQA sweep (all 503 stocks)
foreach ($B in 0.0, 0.5, 1.0, 1.5, 2.0) {
    Write-Host "beta = $B"
    python scripts/train_beta_features_baselines_d3.py --beta $B
}

Write-Host "`nALL DONE" -ForegroundColor Green