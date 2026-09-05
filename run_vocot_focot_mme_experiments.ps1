$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Python = if ($env:FOCOT_PYTHON) { $env:FOCOT_PYTHON } else { "python" }

$RunDir = Join-Path $Root "outputs\vocot_focot_mme_run"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$LogPath = Join-Path $RunDir "run.log"

function Write-RunLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $LogPath -Append
}

$Methods = @(
    @{ Name = "VOCOT"; Key = "vocot"; Out = "outputs\mme_qwen35_9b_vocot_full" },
    @{ Name = "FOCOT"; Key = "focot"; Out = "outputs\mme_qwen35_9b_focot_full" }
)

Write-RunLog "Starting VOCOT/FOCOT MME-RealWorld-Lite prediction run."
Write-RunLog "Python: $(& $Python -c ""import sys; print(sys.executable)"")"

foreach ($method in $Methods) {
    Write-RunLog "Starting prediction: $($method.Name)"
    & $Python mme_qwen_experiment.py `
        --data data\MME-RealWorld-Lite.json `
        --model_path models\Qwen3.5-9B `
        --method $method.Key `
        --max_new_tokens 8 `
        --max_pixels 32768 `
        --save_every 1 `
        --resume `
        --output_dir $method.Out 2>&1 | Tee-Object -FilePath $LogPath -Append

    if ($LASTEXITCODE -ne 0) {
        Write-RunLog "FAILED prediction: $($method.Name) with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-RunLog "Finished prediction: $($method.Name)"
}

Write-RunLog "Starting API judge for VOCOT/FOCOT."
& $Python api_judge_mme_table.py `
    --method_file `
        VOCOT=outputs\mme_qwen35_9b_vocot_full\vocot_predictions.json `
        FOCOT=outputs\mme_qwen35_9b_focot_full\focot_predictions.json `
    --judge_model gpt-5.4-mini `
    --output_dir outputs\api_judge_mme_table_vocot_focot `
    --save_every 10 `
    --resume 2>&1 | Tee-Object -FilePath $LogPath -Append

if ($LASTEXITCODE -ne 0) {
    Write-RunLog "FAILED API judge with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
Write-RunLog "Finished API judge."

Write-RunLog "VOCOT/FOCOT MME experiment script finished."
