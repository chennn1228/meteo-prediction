throw 'BLOCKED: this pre-v2 PowerShell queue is archived in legacy and cannot produce official results.'
$root = "D:\Administrator Files\SEU\12.Project\260801_electricity market\meteo prediction"
$py = Join-Path $root ".venv\Scripts\python.exe"
$trainer = Join-Path $root "src\s03_models\train\train_deep.py"
$logdir = Join-Path $root "reports\03_modeling\00_logs\quantile"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$models = @("mlp","lstm","cnn","tcn","transformer","autoformer","informer","fedformer",
            "itransformer","patchtst","dlinear","timesnet","tsmixer","pinn")
$vnum = @{mlp=2;lstm=3;cnn=4;tcn=5;transformer=6;autoformer=7;informer=8;fedformer=9;
          itransformer=10;patchtst=11;dlinear=12;timesnet=13;tsmixer=14;pinn=15}
$all = New-Object System.Collections.ArrayList
foreach ($t in @("ghi","cloud")) {
  foreach ($m in $models) { [void]$all.Add([pscustomobject]@{target=$t;model=$m}) }
}
$buckets = @(@(),@(),@())
for ($i=0;$i -lt $all.Count;$i++){ $buckets[$i%3] += ,$all[$i] }
$jobScript = {
  param($root,$py,$trainer,$logdir,$bucket,$bid,$vnum)
  foreach($j in $bucket){
    $name="$($j.target)_v$($j.model)"
    $summary=Join-Path $root "reports\03_modeling\v$($vnum[$j.model])_$($j.model)\lite\quantile\$($j.target)\summary.md"
    if(Test-Path -LiteralPath $summary){ "SKIP $name" | Out-File (Join-Path $logdir "$name.log") -Append; continue }
    $log=Join-Path $logdir "$name.log"
    "START $name $(Get-Date -Format s)" | Out-File -FilePath $log -Append
    & $py $trainer --target $j.target --model $j.model --quantile `
        --epochs 2 --subsample 8 --val-subsample 8 --seq-len 24 *>> $log
    "EXIT $LASTEXITCODE $name $(Get-Date -Format s)" | Out-File -FilePath $log -Append
  }
  "BUCKET $bid DONE $(Get-Date -Format s)" | Out-File (Join-Path $logdir "bucket_$bid.done") -Append
}
$jobs=@()
for($i=0;$i -lt 3;$i++){ $jobs += Start-Job -ScriptBlock $jobScript -ArgumentList $root,$py,$trainer,$logdir,$buckets[$i],$i,$vnum }
Wait-Job -Job $jobs | Out-Null
Receive-Job -Job $jobs | Out-Null
"DEEP QUANTILE DONE $(Get-Date -Format s)" | Out-File (Join-Path $logdir "all.done") -Append
