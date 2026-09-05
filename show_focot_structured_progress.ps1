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

$PredPath = "outputs\mme_qwen35_9b_focot_structured_v3_anchor_full\focot_predictions.json"
$ProgressPath = "outputs\mme_qwen35_9b_focot_structured_v3_anchor_full\focot_progress.json"
$LogPath = "outputs\focot_structured_mme_run\run.log"

"Structured FOCOT progress - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
""

$count = 0
$acc = $null
$detail = ""
if (Test-Path $PredPath) {
    try {
        $rows = Get-Content $PredPath -Raw | ConvertFrom-Json
        $count = $rows.Count
        if ($count -gt 0) {
            $correct = @($rows | Where-Object { $_.correct -eq $true }).Count
            $acc = "{0:N2}%" -f (100.0 * $correct / $count)
        }
    } catch {
        $count = 0
    }
}
if (Test-Path $ProgressPath) {
    try {
        $progress = Get-Content $ProgressPath -Raw | ConvertFrom-Json
        if ([int]$progress.completed -gt $count) {
            $count = [int]$progress.completed
        }
        if ($null -ne $progress.accuracy_so_far) {
            $acc = "{0:N2}%" -f (100.0 * [double]$progress.accuracy_so_far)
        }
        if ($progress.status -eq "generating") {
            $detail = " status=generating current=$($progress.current_item_number)/$($progress.total) item=$($progress.current_item_id)"
        } elseif ($progress.status -eq "loading_model") {
            $detail = " status=loading_model"
        } else {
            $eta = if ($null -ne $progress.eta_seconds_this_run) { "{0:N1} min" -f ([double]$progress.eta_seconds_this_run / 60.0) } else { "n/a" }
            $detail = " status=$($progress.status) last=$($progress.last_item_id) pred=$($progress.last_prediction) ans=$($progress.last_answer) eta_this_run=$eta"
        }
    } catch {
        $detail = ""
    }
}

$pct = "{0:N2}%" -f (100.0 * $count / $Total)
if ($acc) {
    "FOCOT-structured {0,4}/{1} ({2}) acc_so_far={3}{4}" -f $count, $Total, $pct, $acc, $detail
} else {
    "FOCOT-structured {0,4}/{1} ({2}){3}" -f $count, $Total, $pct, $detail
}

""
"Running Python processes:"
Get-Process | Where-Object { $_.ProcessName -like "python*" } | Select-Object ProcessName,Id,CPU,WS,StartTime | Format-Table -AutoSize

if (Test-Path $LogPath) {
    ""
    "Last 20 log lines:"
    Get-Content $LogPath -Tail 20
}
