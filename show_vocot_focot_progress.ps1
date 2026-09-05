$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$DataPath = "data\MME-RealWorld-Lite.json"
$Total = 1919
if (Test-Path $DataPath) {
    try {
        $Total = (Get-Content $DataPath -Raw | ConvertFrom-Json).Count
    } catch {
        $Total = 1919
    }
}

$Methods = @(
    @{
        Name = "VOCOT"
        Out = "outputs\mme_qwen35_9b_vocot_full\vocot_predictions.json"
        Progress = "outputs\mme_qwen35_9b_vocot_full\vocot_progress.json"
    },
    @{
        Name = "FOCOT"
        Out = "outputs\mme_qwen35_9b_focot_full\focot_predictions.json"
        Progress = "outputs\mme_qwen35_9b_focot_full\focot_progress.json"
    }
)

"VOCOT/FOCOT progress - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
""

foreach ($method in $Methods) {
    $count = 0
    $acc = $null
    $detail = ""
    if (Test-Path $method.Out) {
        try {
            $rows = Get-Content $method.Out -Raw | ConvertFrom-Json
            $count = $rows.Count
            if ($count -gt 0) {
                $correct = @($rows | Where-Object { $_.correct -eq $true }).Count
                $acc = "{0:N2}%" -f (100.0 * $correct / $count)
            }
        } catch {
            $count = 0
        }
    }
    if (Test-Path $method.Progress) {
        try {
            $progress = Get-Content $method.Progress -Raw | ConvertFrom-Json
            $progressCount = [int]$progress.completed
            if ($progressCount -gt $count) {
                $count = $progressCount
            }
            if ($null -ne $progress.accuracy_so_far) {
                $acc = "{0:N2}%" -f (100.0 * [double]$progress.accuracy_so_far)
            }
            $eta = if ($null -ne $progress.eta_seconds_this_run) { "{0:N1} min" -f ([double]$progress.eta_seconds_this_run / 60.0) } else { "n/a" }
            if ($progress.status -eq "generating") {
                $detail = " status=generating current=$($progress.current_item_number)/$($progress.total) item=$($progress.current_item_id)"
            } elseif ($progress.status -eq "loading_model") {
                $detail = " status=loading_model"
            } else {
                $detail = " status=$($progress.status) last=$($progress.last_item_id) pred=$($progress.last_prediction) ans=$($progress.last_answer) eta_this_run=$eta"
            }
        } catch {
            $detail = ""
        }
    }

    $pct = "{0:N2}%" -f (100.0 * $count / $Total)
    if ($acc) {
        "{0,-8} {1,4}/{2} ({3}) acc_so_far={4}{5}" -f $method.Name, $count, $Total, $pct, $acc, $detail
    } else {
        "{0,-8} {1,4}/{2} ({3})" -f $method.Name, $count, $Total, $pct
    }
}

""
"Running Python processes:"
Get-Process | Where-Object { $_.ProcessName -like "python*" } | Select-Object ProcessName,Id,CPU,WS,StartTime | Format-Table -AutoSize

$LogPath = "outputs\vocot_focot_mme_run\run.log"
if (Test-Path $LogPath) {
    ""
    "Last 20 log lines:"
    Get-Content $LogPath -Tail 20
}
