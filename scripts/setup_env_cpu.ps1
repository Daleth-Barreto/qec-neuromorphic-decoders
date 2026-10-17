# Crea el entorno conda 'qecbb' (Python 3.10) e instala dependencias CPU.
# Uso:  powershell -ExecutionPolicy Bypass -File scripts\setup_env_cpu.ps1
$ErrorActionPreference = "Stop"

$envName = "qecbb"
# Resolve the conda base prefix dynamically instead of hardcoding a user path.
$condaBase = (conda info --base).Trim()
$py = Join-Path $condaBase "envs\$envName\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "[1/4] Creando entorno conda '$envName' (Python 3.10)..."
    conda create -y -n $envName python=3.10 pip
} else {
    Write-Host "[1/4] Entorno '$envName' ya existe."
}

Write-Host "[2/4] Instalando torch CPU (pequeno)..."
& $py -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu

Write-Host "[3/4] Instalando resto de dependencias..."
& $py -m pip install --quiet numpy pandas matplotlib scikit-learn networkx ipykernel torch-geometric ldpc bposd stim huggingface_hub amaranth amaranth-yosys yowasp-yosys yowasp-nextpnr-ice40

Write-Host "[4/4] Registrando kernel Jupyter..."
& $py -m ipykernel install --user --name $envName --display-name "qecbb (py3.10)"

Write-Host "Listo. Verificar con: & `"$py`" scripts\verify_environment.py --device cpu"