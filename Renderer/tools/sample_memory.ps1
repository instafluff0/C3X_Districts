param([string]$Out,[int]$Seconds=300,[int]$IntervalMs=2000)
# Every interval (default 2 s): private bytes, working set and GPU dedicated/shared usage of Civ III
# and the Renderer64 helper, plus system available memory (MB). utc_ms aligns samples with the
# helper trace's usage-session utc_unix_ms. Summarize with Renderer/tools/memory_peak_report.py.
# In the Parallels VM every D3D allocation is charged to the helper's private bytes.
"t,process,pid,private_mb,working_mb,gpu_dedicated_mb,gpu_shared_mb,available_mb,utc_ms" | Out-File -Encoding ascii $Out
$end=(Get-Date).AddSeconds($Seconds);$start=Get-Date
while((Get-Date) -lt $end){
  $now=Get-Date
  $t=[int]($now-$start).TotalSeconds
  $utc=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
  $available=(Get-Counter '\Memory\Available MBytes' -ErrorAction SilentlyContinue).CounterSamples[0].CookedValue
  foreach($name in 'Civ3Conquests','C3XRendererHelper64'){
    foreach($p in Get-Process -Name $name -ErrorAction SilentlyContinue){
      $dedicated=0;$shared=0
      try{
        $samples=(Get-Counter "\GPU Process Memory(pid_$($p.Id)_*)\Dedicated Usage","\GPU Process Memory(pid_$($p.Id)_*)\Shared Usage" -ErrorAction Stop).CounterSamples
        foreach($s in $samples){if($s.Path -like '*dedicated usage'){$dedicated+=$s.CookedValue}else{$shared+=$s.CookedValue}}
      }catch{}
      "{0},{1},{2},{3:F0},{4:F0},{5:F0},{6:F0},{7:F0},{8}" -f $t,$name,$p.Id,($p.PrivateMemorySize64/1MB),($p.WorkingSet64/1MB),($dedicated/1MB),($shared/1MB),$available,$utc | Out-File -Append -Encoding ascii $Out
    }
  }
  $spent=((Get-Date)-$now).TotalMilliseconds
  if($spent -lt $IntervalMs){Start-Sleep -Milliseconds ($IntervalMs-$spent)}
}
