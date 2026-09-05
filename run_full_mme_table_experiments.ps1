$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$RunDir = Join-Path $Root "outputs\full_mme_table_run"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$LogPath = Join-Path $RunDir "run.log"

function Write-RunLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $LogPath -Append
}

$Methods = @(
    @{ Name = "Vanilla";       Key = "vanilla";       Out = "outputs\mme_qwen35_9b_vanilla_full" },
    @{ Name = "Standard_CoT";  Key = "standard_cot";  Out = "outputs\mme_qwen35_9b_standard_cot_full" },
    @{ Name = "Self_Cons_CoT"; Key = "self_cons_cot"; Out = "outputs\mme_qwen35_9b_self_cons_cot_full" },
    @{ Name = "Standard_FOT";  Key = "standard_fot";  Out = "outputs\mme_qwen35_9b_standard_fot_full" },
    @{ Name = "VOCOT";         Key = "vocot";         Out = "outputs\mme_qwen35_9b_vocot_full" },
    @{ Name = "FOCOT";         Key = "focot";         Out = "outputs\mme_qwen35_9b_focot_full" }
)

Write-RunLog "Starting full MME-RealWorld-Lite prediction run."
Write-RunLog "Python: $(python -c ""import sys; print(sys.executable)"")"

foreach ($method in $Methods) {
    Write-RunLog "Starting prediction: $($method.Name)"
    python mme_qwen_experiment.py `
        --data data\MME-RealWorld-Lite.json `
        --model_path models\Qwen3.5-9B `
        --method $method.Key `
        --max_new_tokens 8 `
        --max_pixels 32768 `
        --save_every 10 `
        --resume `
        --output_dir $method.Out 2>&1 | Tee-Object -FilePath $LogPath -Append

    if ($LASTEXITCODE -ne 0) {
        Write-RunLog "FAILED prediction: $($method.Name) with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-RunLog "Finished prediction: $($method.Name)"
}

if ($env:OPENAI_API_KEY) {
    Write-RunLog "OPENAI_API_KEY found. Starting API judge."
    python api_judge_mme_table.py `
        --method_file `
            Vanilla=outputs\mme_qwen35_9b_vanilla_full\vanilla_predictions.json `
            Standard_CoT=outputs\mme_qwen35_9b_standard_cot_full\standard_cot_predictions.json `
            Self_Cons_CoT=outputs\mme_qwen35_9b_self_cons_cot_full\self_cons_cot_predictions.json `
            Standard_FOT=outputs\mme_qwen35_9b_standard_fot_full\standard_fot_predictions.json `
            VOCOT=outputs\mme_qwen35_9b_vocot_full\vocot_predictions.json `
            FOCOT=outputs\mme_qwen35_9b_focot_full\focot_predictions.json `
        --judge_model gpt-5.4-mini `
        --output_dir outputs\api_judge_mme_table_full `
        --save_every 10 `
        --resume 2>&1 | Tee-Object -FilePath $LogPath -Append

    if ($LASTEXITCODE -ne 0) {
        Write-RunLog "FAILED API judge with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-RunLog "Finished API judge."
} else {
    Write-RunLog "OPENAI_API_KEY is not set. Skipping API judge for now."
    Write-RunLog "After predictions finish, set OPENAI_API_KEY and run api_judge_mme_table.py manually or rerun this script with --resume behavior."
}

Write-RunLog "Full MME table experiment script finished."
