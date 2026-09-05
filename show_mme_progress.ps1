$ErrorActionPreference = "SilentlyContinue"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Total = 1919
$Methods = @(
    @{ Name = "Vanilla";       Key = "vanilla";       Out = "outputs\mme_qwen35_9b_vanilla_full" },
    @{ Name = "Standard_CoT";  Key = "standard_cot";  Out = "outputs\mme_qwen35_9b_standard_cot_full" },
    @{ Name = "Self_Cons_CoT"; Key = "self_cons_cot"; Out = "outputs\mme_qwen35_9b_self_cons_cot_full" },
    @{ Name = "Standard_FOT";  Key = "standard_fot";  Out = "outputs\mme_qwen35_9b_standard_fot_full" },
    @{ Name = "VOCOT";         Key = "vocot";         Out = "outputs\mme_qwen35_9b_vocot_full" },
    @{ Name = "FOCOT";         Key = "focot";         Out = "outputs\mme_qwen35_9b_focot_full" }
)

"MME experiment progress - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
""

foreach ($method in $Methods) {
    $path = Join-Path $method.Out "$($method.Key)_predictions.json"
    $count = 0
    $acc = ""
    if (Test-Path $path) {
        try {
            $rows = Get-Content $path -Raw | ConvertFrom-Json
            $count = @($rows).Count
            if ($count -gt 0) {
                $correct = @($rows | Where-Object { $_.correct -eq $true }).Count
                $acc = " acc_so_far={0:N2}%" -f (($correct / $count) * 100)
            }
        } catch {
            $acc = " reading..."
        }
    }
    $pct = "{0:N2}%" -f (($count / $Total) * 100)
    "{0,-14} {1,4}/{2} ({3}){4}" -f $method.Name, $count, $Total, $pct, $acc
}

""
"Running Python processes:"
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, CPU, WS, StartTime | Format-Table -AutoSize | Out-String

$log = "outputs\full_mme_table_run\run.log"
if (Test-Path $log) {
    "Last 20 log lines:"
    Get-Content $log -Tail 20
} else {
    "No run log yet."
}
