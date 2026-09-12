param(
    [string]$UserName = "fahimsiddiqui",
    [string]$NotebookSlug = "dermascanai-phase-2-convnext-tiny-baseline",
    [int]$NotebookId = 131967226
)

$ErrorActionPreference = "Stop"
$token = [Environment]::GetEnvironmentVariable("KAGGLE_API_TOKEN", "User")
if (-not $token) { throw "KAGGLE_API_TOKEN is not configured." }

$sourceFiles = @(
    "preprocessing.py",
    "skin_dataset.py",
    "dataset_preparation.py",
    "verify_dataset.py",
    "convnext_baseline.py",
    "train.py",
    "evaluate.py",
    "requirements.txt"
)
$encoded = [ordered]@{}
foreach ($name in $sourceFiles) {
    $path = Join-Path $PSScriptRoot $name
    $encoded[$name] = [Convert]::ToBase64String([IO.File]::ReadAllBytes($path))
}
$encodedJson = $encoded | ConvertTo-Json -Compress

$setupSource = @"
from pathlib import Path
import base64, json, os, subprocess, sys

# Kaggle's current default CUDA build does not contain a Tesla T4 (sm_75)
# kernel image. Install a stable official CUDA wheel that supports the T4.
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade',
    'torch==2.7.1', 'torchvision==0.22.1',
    '--index-url', 'https://download.pytorch.org/whl/cu126',
], check=True)

import torch

WORK_DIR = Path('/kaggle/working/dermascanai')
WORK_DIR.mkdir(parents=True, exist_ok=True)
encoded_files = json.loads(r'''$encodedJson''')
for name, content in encoded_files.items():
    (WORK_DIR / name).write_bytes(base64.b64decode(content))
os.chdir(WORK_DIR)
print('Clean-room Phase 1/2 code written to:', WORK_DIR)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('Compute capability:', torch.cuda.get_device_capability(0))
    print('Torch CUDA architectures:', torch.cuda.get_arch_list())
print('PyTorch:', torch.__version__, 'Torchvision will load in subprocesses')
print('Files:', sorted(encoded_files))
"@

$verifySource = @"
import subprocess, sys
subprocess.run([
    sys.executable, 'verify_dataset.py',
    '--dataset-root', '/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000',
    '--artifacts-dir', 'artifacts',
    '--overwrite-splits',
], check=True)
"@

$trainSource = @"
import subprocess, sys
subprocess.run([
    sys.executable, 'train.py',
    '--dataset-root', '/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000',
    '--artifacts-dir', 'artifacts',
    '--output-dir', 'outputs/convnext_tiny',
    '--epochs', '30', '--batch-size', '32', '--num-workers', '2',
    '--learning-rate', '3e-4', '--weight-decay', '1e-4',
    '--early-stopping-patience', '7', '--scheduler-patience', '2',
    '--seed', '42', '--device', 'cuda',
], check=True)
"@

$evaluateSource = @"
import subprocess, sys
subprocess.run([
    sys.executable, 'evaluate.py',
    '--dataset-root', '/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000',
    '--artifacts-dir', 'artifacts',
    '--checkpoint', 'outputs/convnext_tiny/best_convnext_tiny.pt',
    '--output-dir', 'outputs/convnext_tiny/test',
    '--batch-size', '32', '--num-workers', '2',
    '--seed', '42', '--device', 'cuda',
], check=True)

from pathlib import Path
print('Final output files:')
for path in sorted(Path('outputs/convnext_tiny').rglob('*')):
    if path.is_file(): print(path, path.stat().st_size)
"@

function New-CodeCell([string]$source) {
    return [ordered]@{
        cell_type = "code"
        execution_count = $null
        metadata = @{}
        outputs = @()
        source = $source
    }
}
$notebook = [ordered]@{
    cells = @(
        (New-CodeCell $setupSource),
        (New-CodeCell $verifySource),
        (New-CodeCell $trainSource),
        (New-CodeCell $evaluateSource)
    )
    metadata = @{
        kernelspec = @{display_name="Python 3"; language="python"; name="python3"}
        language_info = @{name="python"; version="3.12"}
    }
    nbformat = 4
    nbformat_minor = 5
}
$notebookText = $notebook | ConvertTo-Json -Depth 20 -Compress
$request = @{
    slug = "$UserName/$NotebookSlug"
    newTitle = "DermaScanAI Phase 2 ConvNeXt-Tiny Baseline"
    text = $notebookText
    language = "python"
    kernelType = "notebook"
    isPrivate = $true
    enableGpu = $true
    enableTpu = $false
    enableInternet = $true
    datasetDataSources = @(
        "fahimsiddiqui/dermascanai-code",
        "kmader/skin-cancer-mnist-ham10000"
    )
}
$body = $request | ConvertTo-Json -Depth 30 -Compress
$headers = @{Authorization="Bearer $token"; Accept="application/json"}
# Kaggle MCP currently drops datasetDataSources in save_notebook. The official
# kernels push endpoint preserves them; MCP remains responsible for monitoring.
$response = Invoke-WebRequest -Uri "https://www.kaggle.com/api/v1/kernels/push" `
    -Method Post -Headers $headers -ContentType "application/json" -Body $body
$response.Content
