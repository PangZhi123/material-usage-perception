param(
    [int]$Port = 8001,
    [string]$BindAddress = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$projectDirectory = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectDirectory
$env:PYTHONPATH = Join-Path $projectDirectory "src"
if (-not $env:MATERIAL_PERCEPTION_MODEL_PATH) {
    $env:MATERIAL_PERCEPTION_MODEL_PATH = Join-Path $projectDirectory "models/best.pt"
}
if (-not $env:MATERIAL_PERCEPTION_MODEL_VERSION) {
    $env:MATERIAL_PERCEPTION_MODEL_VERSION = "local-test-best"
}
if (-not $env:MATERIAL_PERCEPTION_LOCAL_VIDEO_ROOT) {
    $env:MATERIAL_PERCEPTION_LOCAL_VIDEO_ROOT = Join-Path $projectDirectory "test-data"
}
if (-not $env:YOLO_CONFIG_DIR) {
    $env:YOLO_CONFIG_DIR = Join-Path $projectDirectory "outputs/ultralytics"
}
Write-Host "Test console: http://127.0.0.1:$Port/test-console/"
python -m uvicorn material_usage_perception.api.app:app --host $BindAddress --port $Port
exit $LASTEXITCODE
