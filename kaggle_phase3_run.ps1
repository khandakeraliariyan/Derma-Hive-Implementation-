param(
    [string]$UserName = "fahimsiddiqui",
    [string]$NotebookSlug = "dermascanai-phase-3-skinnet-hda"
)

$ErrorActionPreference = "Stop"
$token = [Environment]::GetEnvironmentVariable("KAGGLE_API_TOKEN", "User")
if (-not $token) { throw "KAGGLE_API_TOKEN is not configured." }

$sourceFiles = @(
    "preprocessing.py", "skin_dataset.py", "dataset_preparation.py",
    "convnext_baseline.py", "skinnet_hda.py", "train_skinnet_hda.py",
    "evaluate_skinnet_hda.py", "verify_skinnet_hda.py",
    "measure_skinnet_efficiency.py", "requirements.txt",
    "requirements_phase3.txt"
)
$encoded = [ordered]@{}
foreach ($name in $sourceFiles) {
    $path = Join-Path $PSScriptRoot $name
    $encoded[$name] = [Convert]::ToBase64String([IO.File]::ReadAllBytes($path))
}
$splitPath = "C:\tmp\dermascanai_phase3\ham10000_splits.csv"
if (-not (Test-Path -LiteralPath $splitPath -PathType Leaf)) {
    throw "Exact saved Phase 1 split is missing: $splitPath"
}
$splitBytes = [IO.File]::ReadAllBytes($splitPath)
$compressedStream = New-Object IO.MemoryStream
$gzipStream = New-Object IO.Compression.GZipStream(
    $compressedStream, [IO.Compression.CompressionMode]::Compress, $true
)
$gzipStream.Write($splitBytes, 0, $splitBytes.Length)
$gzipStream.Dispose()
$encoded["artifacts/ham10000_splits.csv.gz"] = [Convert]::ToBase64String(
    $compressedStream.ToArray()
)
$compressedStream.Dispose()
$encodedJson = $encoded | ConvertTo-Json -Compress

$setupSource = @"
from pathlib import Path
import base64, gzip, json, os, subprocess, sys

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade',
    'torch==2.7.1', 'torchvision==0.22.1',
    '--index-url', 'https://download.pytorch.org/whl/cu126',
], check=True)
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
    'fvcore>=0.1.5.post20221221',
], check=True)

WORK_DIR = Path('/kaggle/working/dermascanai')
WORK_DIR.mkdir(parents=True, exist_ok=True)
encoded_files = json.loads(r'''$encodedJson''')
for name, content in encoded_files.items():
    destination = WORK_DIR / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(base64.b64decode(content))
compressed_split = WORK_DIR / 'artifacts/ham10000_splits.csv.gz'
(WORK_DIR / 'artifacts/ham10000_splits.csv').write_bytes(
    gzip.decompress(compressed_split.read_bytes())
)
compressed_split.unlink()
os.chdir(WORK_DIR)

import pandas as pd, torch
split = pd.read_csv('artifacts/ham10000_splits.csv')
assert len(split) == 10015
assert split['split'].value_counts().to_dict() == {'train': 7002, 'val': 1508, 'test': 1505}
metadata_files = list(Path('/kaggle/input').rglob('HAM10000_metadata.csv'))
if len(metadata_files) != 1:
    raise RuntimeError(f'Expected one HAM10000_metadata.csv; found: {metadata_files}')
dataset_root = metadata_files[0].parent.resolve()
(WORK_DIR / 'dataset_root.txt').write_text(str(dataset_root), encoding='utf-8')
print('Clean-room Phase 3 code written to:', WORK_DIR)
print('Exact Phase 1 split restored:', split['split'].value_counts().to_dict())
print('Unique lesions:', split['lesion_id'].nunique())
print('Discovered HAM10000 root:', dataset_root)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('Compute capability:', torch.cuda.get_device_capability(0))
print('PyTorch:', torch.__version__)
print('Files:', sorted(encoded_files))
"@

$verifySource = @"
import os, subprocess, sys
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
dataset_root = open('dataset_root.txt', encoding='utf-8').read().strip()
subprocess.run([
    sys.executable, 'verify_skinnet_hda.py',
    '--dataset-root', dataset_root,
    '--artifacts-dir', 'artifacts', '--batch-size', '2',
    '--num-workers', '2', '--device', 'cuda',
], check=True)
"@

$trainSource = @"
import os, subprocess, sys
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
dataset_root = open('dataset_root.txt', encoding='utf-8').read().strip()
subprocess.run([
    sys.executable, 'train_skinnet_hda.py',
    '--dataset-root', dataset_root,
    '--artifacts-dir', 'artifacts', '--output-dir', 'outputs/skinnet_hda',
    '--epochs', '30', '--batch-size', '32',
    '--learning-rate', '1e-4', '--minimum-learning-rate', '1e-6',
    '--weight-decay', '1e-4', '--early-stopping-patience', '5',
    '--num-workers', '2', '--seed', '42', '--device', 'cuda',
], check=True)
"@

$evaluateSource = @"
import os, subprocess, sys
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
dataset_root = open('dataset_root.txt', encoding='utf-8').read().strip()
subprocess.run([
    sys.executable, 'evaluate_skinnet_hda.py',
    '--dataset-root', dataset_root,
    '--artifacts-dir', 'artifacts',
    '--checkpoint', 'outputs/skinnet_hda/best_skinnet_hda.pt',
    '--output-dir', 'outputs/skinnet_hda/test', '--batch-size', '32',
    '--num-workers', '2', '--seed', '42', '--device', 'cuda',
], check=True)
subprocess.run([
    sys.executable, 'measure_skinnet_efficiency.py',
    '--checkpoint', 'outputs/skinnet_hda/best_skinnet_hda.pt',
    '--output', 'outputs/skinnet_hda/efficiency.json', '--device', 'cuda',
    '--warmup-runs', '30', '--timed-runs', '100',
], check=True)

from pathlib import Path
print('Final Phase 3 output files:')
for path in sorted(Path('outputs/skinnet_hda').rglob('*')):
    if path.is_file(): print(path, path.stat().st_size)
"@

function New-CodeCell([string]$source) {
    return [ordered]@{
        cell_type = "code"; execution_count = $null; metadata = @{};
        outputs = @(); source = $source
    }
}
$notebook = [ordered]@{
    cells = @(
        (New-CodeCell $setupSource), (New-CodeCell $verifySource),
        (New-CodeCell $trainSource), (New-CodeCell $evaluateSource)
    )
    metadata = @{
        kernelspec = @{display_name="Python 3"; language="python"; name="python3"}
        language_info = @{name="python"; version="3.12"}
    }
    nbformat = 4; nbformat_minor = 5
}
$request = @{
    slug = "$UserName/$NotebookSlug"
    newTitle = "DermaScanAI Phase 3 SkinNet-HDA"
    text = ($notebook | ConvertTo-Json -Depth 20 -Compress)
    language = "python"; kernelType = "notebook"; isPrivate = $true
    enableGpu = $true; enableTpu = $false; enableInternet = $true
    datasetDataSources = @("kmader/skin-cancer-mnist-ham10000")
}
$headers = @{Authorization="Bearer $token"; Accept="application/json"}
$response = Invoke-WebRequest -Uri "https://www.kaggle.com/api/v1/kernels/push" `
    -Method Post -Headers $headers -ContentType "application/json" `
    -Body ($request | ConvertTo-Json -Depth 30 -Compress)
$response.Content
