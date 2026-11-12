[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)][string]$RemotePath,
    [string]$RemoteHost = 'daoqm@slurm.csse.rose-hulman.edu',
    [switch]$IncludeData
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($RemoteHost -notmatch '^[A-Za-z0-9_.@-]+$' -or $RemoteHost.StartsWith('-')) {
    throw 'RemoteHost must be a simple SSH host or user@host.'
}
if ($RemotePath -notmatch '^/?[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$' -or
    ($RemotePath -split '/') -contains '..' -or ($RemotePath -split '/') -contains '.' -or
    $RemotePath.StartsWith('-')) {
    throw 'RemotePath must be an explicit project path without dot segments, spaces or shell expressions.'
}
$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceNames = @(
    'amodal_benchmark', 'configs', 'scripts', 'tests', 'cookbooks', 'docs', 'requirements',
    'evaluate_open_world.py', 'evaluate_pdmc_open_world.py', 'prepare_open_world.py',
    'prepare_visible_references.py', 'compare_open_world.py', 'run_open_world_controls.py',
    'run_amodal_slurm.sh', 'run_pdmc_open_world.sh', 'pyproject.toml', 'requirements.txt',
    'MANIFEST.in', '.python-version', 'CONTRIBUTING.md', 'BENCHMARK_WORKFLOW.md', 'requirements-analysis.txt',
    'README.md', 'SETUP_NEW_COMPETITION.md', 'OPEN_WORLD_REPRODUCTION.md', 'AMODAL_SLURM.md',
    '.gitignore', '.gitattributes'
)
if ($IncludeData) { $sourceNames += 'datasets/open_world_amodal' }
# Build a clean staging tree: even a recursive source upload must exclude Python
# caches, current notes, credentials, output artifacts and the local ledger.
$files = foreach ($name in $sourceNames) {
    $itemPath = Join-Path $projectRoot $name
    if (-not (Test-Path -LiteralPath $itemPath)) { throw "Missing upload source: $name" }
    $item = Get-Item -LiteralPath $itemPath
    if ($item.PSIsContainer) {
        Get-ChildItem -LiteralPath $itemPath -Recurse -File -Force |
            Where-Object { $_.FullName -notmatch '[\\/]__pycache__[\\/]|\.py[co]$' }
    } else { $item }
}
$remoteDestination = "${RemoteHost}:${RemotePath}/"
Write-Output "Source files: $($files.Count); destination: $remoteDestination"
if ($PSCmdlet.ShouldProcess($remoteDestination, 'Create project directories and upload selected source files')) {
    $staging = Join-Path ([System.IO.Path]::GetTempPath()) ('amodal-upload-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $staging | Out-Null
    try {
        foreach ($file in $files) {
            $relative = $file.FullName.Substring($projectRoot.Length + 1)
            $target = Join-Path $staging $relative
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
            Copy-Item -LiteralPath $file.FullName -Destination $target
        }
        & ssh $RemoteHost "mkdir -p -- '$RemotePath/logs'"
        if ($LASTEXITCODE -ne 0) { throw "Remote directory creation failed ($LASTEXITCODE)." }
        $items = @(Get-ChildItem -LiteralPath $staging -Force | ForEach-Object { $_.FullName })
        & scp -r @items $remoteDestination
        if ($LASTEXITCODE -ne 0) { throw "Source upload failed ($LASTEXITCODE)." }
    } finally {
        $resolvedStaging = [System.IO.Path]::GetFullPath($staging)
        $tempRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
        if (-not $resolvedStaging.StartsWith($tempRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
            (Split-Path -Leaf $resolvedStaging) -notmatch '^amodal-upload-[a-f0-9]{32}$') {
            throw 'Refusing cleanup outside the verified upload staging directory.'
        }
        Remove-Item -LiteralPath $resolvedStaging -Recurse -Force
    }
}
