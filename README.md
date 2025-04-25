# My Orchestra
For the python environment and exe running on the windows side, please install Fluidsynth first:


```powershell
$downloadUrl = "https://github.com/FluidSynth/fluidsynth/releases/download/v2.4.4/fluidsynth-2.4.4-win10-x64.zip"
$outputZip = "C:\tools\fluidsynth.zip"
$extractPath = "C:\tools"

if (!(Test-Path "C:\tools")) {
    New-Item -ItemType Directory -Path "C:\tools"
}

Invoke-WebRequest -Uri $downloadUrl -OutFile $outputZip

Expand-Archive -Path $outputZip -DestinationPath $extractPath -Force

Remove-Item $outputZip

Write-Output "file downloaded to $extractPath"
```

A midi playback system controlled by video stream image recognition🤏🪄

新增.exe执行文件
<img width="1280" alt="Screen Shot 2025-03-10 at 6 46 07 PM" src="https://github.com/user-attachments/assets/9f3439c8-0480-4b59-bcaf-55868248756d" />



