$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Test-Path "streamlit_app.py")) {
    throw "streamlit_app.py was not found in $projectRoot"
}

# Keep runtime stable on Windows when protobuf/transformers stacks are present.
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"
$env:USE_TF = "0"
$env:TRANSFORMERS_NO_TF = "1"

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$defaultPython = "C:\Program Files\Python310\python.exe"

$pythonCandidates = @()
if (Test-Path $venvPython) {
    $pythonCandidates += $venvPython
}
if (Test-Path $defaultPython) {
    $pythonCandidates += $defaultPython
}
$pythonCandidates += "python"

$pythonExe = $null
foreach ($candidate in $pythonCandidates) {
    try {
        & $candidate -c "import streamlit" *> $null
        if ($LASTEXITCODE -eq 0) {
            $pythonExe = $candidate
            break
        }
    } catch {
        # Try next Python candidate.
    }
}

if (-not $pythonExe) {
    throw "No Python interpreter with streamlit installed was found. Run pip install -r requirements.txt."
}

if (-not (Test-Path ".env")) {
    Write-Warning "No .env file found. Create one from .env.example and set GOOGLE_API_KEY."
}

Write-Host "Using Python:" $pythonExe
Write-Host "Starting Streamlit app on http://localhost:8501"

& $pythonExe -m streamlit run streamlit_app.py
