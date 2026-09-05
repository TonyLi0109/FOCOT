$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Python = if ($env:FOCOT_PYTHON) { $env:FOCOT_PYTHON } else { "python" }

$RunDir = Join-Path $Root "outputs\focot_forest128_mme_run"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$LogPath = Join-Path $RunDir "run.log"

function Write-RunLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $LogPath -Append
}

Write-RunLog "Starting FOCOT forest-128 MME-RealWorld-Lite prediction run."
Write-RunLog "Python: $(& $Python -c ""import sys; print(sys.executable)"")"

& $Python mme_qwen_experiment.py `
    --data data\MME-RealWorld-Lite.json `
    --model_path models\Qwen3.5-9B `
    --method focot `
    --max_new_tokens 128 `
    --max_pixels 32768 `
    --save_every 1 `
    --resume `
    --output_dir outputs\mme_qwen35_9b_focot_forest128_full 2>&1 | Tee-Object -FilePath $LogPath -Append

if ($LASTEXITCODE -ne 0) {
    Write-RunLog "FAILED prediction: FOCOT forest-128 with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
Write-RunLog "Finished prediction: FOCOT forest-128"

Write-RunLog "Starting API judge for FOCOT forest-128."
& $Python api_judge_mme_table.py `
    --method_file `
        FOCOT=outputs\mme_qwen35_9b_focot_forest128_full\focot_predictions.json `
    --judge_model gpt-5.4-mini `
    --output_dir outputs\api_judge_mme_table_focot_forest128 `
    --save_every 10 `
    --resume 2>&1 | Tee-Object -FilePath $LogPath -Append

if ($LASTEXITCODE -ne 0) {
    Write-RunLog "FAILED API judge with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
Write-RunLog "Finished API judge for FOCOT forest-128."
Write-RunLog "FOCOT forest-128 experiment script finished."
