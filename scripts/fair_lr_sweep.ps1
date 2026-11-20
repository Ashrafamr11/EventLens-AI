# OPTIONAL, NOT YET EXECUTED. Total = 24 new runs (4 models x 2 LR x 3 seeds).
# Execute only if you have time, original LLP data and feature files locally, and enough GPU/storage.
param([Parameter(Mandatory=$true)][string]$ArtifactRoot)
$ErrorActionPreference = 'Stop'
$models = @('baseline','bigru','transformer','cross_modal')
$lrs = @(0.001, 0.0003)
$seeds = @(42,43,44)
foreach ($model in $models) {
    foreach ($lr in $lrs) {
        foreach ($seed in $seeds) {
            python src/train.py --artifact-root $ArtifactRoot --model $model --seed $seed --learning-rate $lr --epochs 5 --batch-size 64 --device auto
            if ($LASTEXITCODE -ne 0) { throw "Training failed: $model LR=$lr seed=$seed" }
        }
    }
}
