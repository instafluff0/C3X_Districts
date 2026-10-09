param([string]$Out,[int]$Seconds=300)
# Every 2 s: private bytes, working set and GPU dedicated/shared usage of Civ III
# and the Renderer64 helper, plus system available memory (MB).
"t,process,pid,private_mb,working_mb,gpu_dedicated_mb,gpu_shared_mb,available_mb" | Out-File -Encoding ascii $Out
$end=(Get-Date).AddSeconds($Seconds);$start=Get-Date
while((Get-Date) -lt $end){
  $t=[int]((Get-Date)-$start).TotalSeconds
  $available=(Get-Counter '\Memory\Available MBytes' -ErrorAction SilentlyContinue).CounterSamples[0].CookedValue
  foreach($name in 'Civ3Conquests','C3XRendererHelper64'){
    foreach($p in Get-Process -Name $name -ErrorAction SilentlyContinue){
      $dedicated=0;$shared=0
      try{
        $samples=(Get-Counter "\GPU Process Memory(pid_$($p.Id)_*)\Dedicated Usage","\GPU Process Memory(pid_$($p.Id)_*)\Shared Usage" -ErrorAction Stop).CounterSamples
        foreach($s in $samples){if($s.Path -like '*dedicated usage'){$dedicated+=$s.CookedValue}else{$shared+=$s.CookedValue}}
      }catch{}
      "{0},{1},{2},{3:F0},{4:F0},{5:F0},{6:F0},{7:F0}" -f $t,$name,$p.Id,($p.PrivateMemorySize64/1MB),($p.WorkingSet64/1MB),($dedicated/1MB),($shared/1MB),$available | Out-File -Append -Encoding ascii $Out
    }
  }
  Start-Sleep -Seconds 2
}
