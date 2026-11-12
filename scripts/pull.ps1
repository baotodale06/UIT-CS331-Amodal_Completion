[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)][string]$RemotePath,
    [Parameter(Mandatory = $true)][string]$Run,
    [string]$RemoteHost = 'daoqm@slurm.csse.rose-hulman.edu',
    [string]$LocalPath,
    [switch]$IncludeLedger
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
if ($Run -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw 'Run must be a single output directory name.' }
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $LocalPath) { $LocalPath = Join-Path $projectRoot "output/$Run" }
$destination = [System.IO.Path]::GetFullPath($LocalPath)
if (Test-Path -LiteralPath $destination) { throw "Destination already exists; choose a new LocalPath: $destination" }
$source = "${RemoteHost}:${RemotePath}/output/$Run"
Write-Output "Download: $source -> $destination"
if ($PSCmdlet.ShouldProcess($destination, 'Download one run and optional separate ledger snapshot')) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    & scp -r $source $destination
    if ($LASTEXITCODE -ne 0) { throw "Run download failed ($LASTEXITCODE); partial files remain at $destination" }
    if ($IncludeLedger) {
        $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
        $ledgerDestination = Join-Path $projectRoot "ledger-snapshots/$stamp"
        New-Item -ItemType Directory -Path $ledgerDestination | Out-Null
        & scp "${RemoteHost}:${RemotePath}/experiments.csv" "${RemoteHost}:${RemotePath}/experiment_report.md" $ledgerDestination
        if ($LASTEXITCODE -ne 0) { throw "Ledger download failed ($LASTEXITCODE); local registry was not changed." }
    }
}
