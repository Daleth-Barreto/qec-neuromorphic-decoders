# Cambia el torch del entorno qecbb a la build CUDA (para entrenamiento robusto con GPU).
# REQUISITO PREVIO: entorno CPU ya creado (scripts\setup_env_cpu.ps1).
# Requiere driver NVIDIA compatible. RTX 3050 -> cu124 recomendado.
# Uso:  powershell -ExecutionPolicy Bypass -File scripts\setup_env_gpu.ps1
$ErrorActionPreference = "Stop"

$envName = "qecbb"
# Resolve the conda base prefix dynamically instead of hardcoding a user path.
$condaBase = (conda info --base).Trim()
$py = Join-Path $condaBase "envs\$envName\python.exe"

if (-not (Test-Path $py)) {
    Write-Error "Entorno '$envName' no existe. Ejecuta primero setup_env_cpu.ps1"
}

Write-Host "[1/3] Desinstalando torch CPU..."
& $py -m pip uninstall -y torch

Write-Host "[2/3] Instalando torch CUDA 12.4 (descarga grande ~2.5 GB)..."
& $py -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124

Write-Host "[3/3] Verificando GPU..."
& $py -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

Write-Host "Listo. Los scripts ya usan '--device cuda' o 'auto' para usar la GPU."