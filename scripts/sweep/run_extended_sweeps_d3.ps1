$ErrorActionPreference = "Continue"

Write-Host "EXTENDED PARAMETER SWEEPS - DATASET 3" -ForegroundColor Cyan

Write-Host "`n>>> [1/4] Extended beta sweep (beta = 3, 4, 5)" -ForegroundColor Yellow
foreach ($B in 3.0, 4.0, 5.0) {
    Write-Host "  beta = $B (m=4, tau=2, rr=0.1)"
    python scripts/train/train_beta_features_baselines_d3.py --beta $B
}

Write-Host "`n>>> [2/4] Embedding dimension m sweep (beta=2.0, tau=2)" -ForegroundColor Yellow
foreach ($M in 2, 3, 5, 6, 7, 8) {
    Write-Host "  m = $M"
    python scripts/train/train_beta_features_baselines_d3.py --beta 2.0 --m $M --tau 2
}

Write-Host "`n>>> [3/4] Time delay tau sweep (beta=2.0, m=4)" -ForegroundColor Yellow
foreach ($T in 1, 3, 4, 5) {
    Write-Host "  tau = $T"
    python scripts/train/train_beta_features_baselines_d3.py --beta 2.0 --m 4 --tau $T
}

Write-Host "`n>>> [4/4] Recurrence rate sweep (beta=2.0, m=4, tau=2)" -ForegroundColor Yellow
foreach ($R in 0.05, 0.20) {
    Write-Host "  rr = $R"
    python scripts/train/train_beta_features_baselines_d3.py --beta 2.0 --m 4 --tau 2 --rr $R
}

Write-Host "`nALL EXTENDED SWEEPS DONE" -ForegroundColor Green
Write-Host @"

Summary of runs:
  - 3 extended beta values (3.0, 4.0, 5.0)
  - 6 m values (2, 3, 5, 6, 7, 8) at beta=2.0
  - 4 tau values (1, 3, 4, 5) at beta=2.0
  - 2 RR values (0.05, 0.20) at beta=2.0
  = 15 total runs (~8 min each = ~2 hours)
"@