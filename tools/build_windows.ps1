$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$unity = 'C:\Program Files\Unity\Hub\Editor\6000.0.45f1\Editor\Unity.exe'
$log = Join-Path $project 'unity-build.log'
& $unity -batchmode -nographics -quit -projectPath $project -executeMethod BuildPresentation.BuildWindows -logFile $log
if ($LASTEXITCODE -ne 0) {
    Get-Content -LiteralPath $log -Tail 70
    throw "Unity build failed with exit code $LASTEXITCODE"
}
Get-Content -LiteralPath $log -Tail 30
