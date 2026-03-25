# run_extended_sweeps_d2.ps1
# Extended parameter sweeps on Dataset 2 - literature-supported values
# Usage: powershell -ExecutionPolicy Bypass -File scripts\run_extended_sweeps_d2.ps1
#
# IMPORTANT: Make sure train_beta_features_baselines.py points to dataset2_ parquets!
#
# Literature references for each sweep:
#   beta > 2: Dreesen et al. 2025 (Table I, tested beta=0..4)
#   m sweep:  Marwan et al. 2007 (Section 3.1), Zbilut & Webber 1992
#   tau sweep: Marwan et al. 2007 (Section 3.1), mutual information criterion
#   RR sweep: Marwan & Kraemer 2024 (Section 2.8, threshold selection)

$ErrorActionPreference = "Continue"

Write-Host "EXTENDED PARAMETER SWEEPS - DATASET 2" -ForegroundColor Cyan

# 1. Extended beta sweep
Write-Host "`n>>> [1/4] Extended beta sweep (beta = 3, 4, 5)" -ForegroundColor Yellow
foreach ($B in 3.0, 4.0, 5.0) {
    Write-Host "  beta = $($B) (m=4, tau=2, rr=0.1)"
    python scripts/train_beta_features_baselines.py --beta $B
}

# 2. m sweep at best beta=0.5
Write-Host "`n>>> [2/4] Embedding dimension m sweep (beta=0.5, tau=2)" -ForegroundColor Yellow
foreach ($M in 2, 3, 5, 6, 7, 8) {
    Write-Host "  m = $($M)"
    python scripts/train_beta_features_baselines.py --beta 0.5 --m $M --tau 2
}

# 3. tau sweep at best beta=0.5
Write-Host "`n>>> [3/4] Time delay tau sweep (beta=0.5, m=4)" -ForegroundColor Yellow
foreach ($T in 1, 3, 4, 5) {
    Write-Host "  tau = $($T)"
    python scripts/train_beta_features_baselines.py --beta 0.5 --m 4 --tau $T
}

# 4. Recurrence rate sweep
Write-Host "`n>>> [4/4] Recurrence rate sweep (beta=0.5, m=4, tau=2)" -ForegroundColor Yellow
foreach ($R in 0.05, 0.20) {
    Write-Host "  rr = $($R)"
    python scripts/train_beta_features_baselines.py --beta 0.5 --m 4 --tau 2 --rr $R
}

Write-Host "`nALL EXTENDED SWEEPS DONE" -ForegroundColor Green

$summary = @"
Summary of runs:
  - 3 extended beta values (3.0, 4.0, 5.0)
  - 6 m values (2, 3, 5, 6, 7, 8) at beta=0.5
  - 4 tau values (1, 3, 4, 5) at beta=0.5
  - 2 RR values (0.05, 0.20) at beta=0.5
  = 15 total runs (~1 minute each = ~15 minutes)
"@

Write-Host $summary