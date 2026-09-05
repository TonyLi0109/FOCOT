$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Python = if ($env:FOCOT_PYTHON) { $env:FOCOT_PYTHON } else { "python" }

$RunDir = Join-Path $Root "outputs\focot_structured_mme_run"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$LogPath = Join-Path $RunDir "run.log"

function Write-RunLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $LogPath -Append
}

Write-RunLog "Starting structured FOCOT MME-RealWorld-Lite prediction run."
Write-RunLog "Python: $(& $Python -c ""import sys; print(sys.executable)"")"

$PredPath = Join-Path $Root "outputs\mme_qwen35_9b_focot_structured_v3_anchor_full\focot_predictions.json"
$Total = 1919
$ChunkSize = 20
$MaxChunks = [Math]::Ceiling($Total / $ChunkSize) + 10

for ($chunk = 1; $chunk -le $MaxChunks; $chunk++) {
    $Done = 0
    if (Test-Path $PredPath) {
        try {
            $Done = (Get-Content $PredPath -Raw | ConvertFrom-Json).Count
        } catch {
            $Done = 0
        }
    }
    if ($Done -ge $Total) {
        Write-RunLog "Structured FOCOT predictions complete: $Done/$Total"
        break
    }

    Write-RunLog "Starting structured FOCOT chunk $chunk with $Done/$Total already complete."
    & $Python mme_qwen_experiment.py `
        --data data\MME-RealWorld-Lite.json `
        --model_path models\Qwen3.5-9B `
        --method focot `
        --focot_mode structured `
        --focot_branch_tokens 40 `
        --anchor_predictions outputs\mme_qwen35_9b_vocot_full\vocot_predictions.json `
        --anchor_weight 0.70 `
        --max_new_tokens 40 `
        --max_pixels 32768 `
        --max_items_per_run $ChunkSize `
        --save_every 1 `
        --resume `
        --output_dir outputs\mme_qwen35_9b_focot_structured_v3_anchor_full 2>&1 | Tee-Object -FilePath $LogPath -Append

    if ($LASTEXITCODE -ne 0) {
        Write-RunLog "FAILED prediction chunk $chunk with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}
Write-RunLog "Finished prediction: structured FOCOT"

Write-RunLog "Starting API judge for structured FOCOT."
& $Python api_judge_mme_table.py `
        --method_file `
        FOCOT=outputs\mme_qwen35_9b_focot_structured_v3_anchor_full\focot_predictions.json `
    --judge_model gpt-5.4-mini `
    --output_dir outputs\api_judge_mme_table_focot_structured_v3_anchor `
    --save_every 10 `
    --resume 2>&1 | Tee-Object -FilePath $LogPath -Append

if ($LASTEXITCODE -ne 0) {
    Write-RunLog "FAILED API judge with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
Write-RunLog "Finished API judge for structured FOCOT."
Write-RunLog "Structured FOCOT experiment script finished."
