[CmdletBinding()]
param(
    [string]$Submission = "ivc_2026-09-30_v4",
    [string]$OutputName = "PACE-Seg_IVC_Manuscript.pdf"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$sourceDir = Join-Path $repoRoot "paper\submission\$Submission"
$sourceTex = Join-Path $sourceDir "main.tex"
$outputDir = Join-Path $repoRoot "output\pdf"
$outputPdf = Join-Path $outputDir $OutputName

if (-not (Test-Path -LiteralPath $sourceTex)) {
    throw "Manuscript source not found: $sourceTex"
}

$tectonicCommand = Get-Command tectonic -ErrorAction SilentlyContinue
if ($tectonicCommand) {
    $tectonic = $tectonicCommand.Source
} else {
    $toolDir = Join-Path $repoRoot ".tools\tectonic"
    $tectonic = Join-Path $toolDir "tectonic.exe"
    if (-not (Test-Path -LiteralPath $tectonic)) {
        $archive = Join-Path $toolDir "tectonic.zip"
        $url = "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-pc-windows-msvc.zip"
        $expectedSha256 = "F61CE51F0B0ADE1015B7DE7EF368541C5424E9756ECBD0D7AF97D6D48030845F"
        New-Item -ItemType Directory -Force -Path $toolDir | Out-Null
        Invoke-WebRequest -Uri $url -OutFile $archive
        $actualSha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
        if ($actualSha256 -ne $expectedSha256) {
            throw "Tectonic archive checksum mismatch: $actualSha256"
        }
        Expand-Archive -LiteralPath $archive -DestinationPath $toolDir -Force
    }
}

Push-Location $sourceDir
try {
    & $tectonic "main.tex" "--keep-logs" "--keep-intermediates"
    if ($LASTEXITCODE -ne 0) {
        throw "Tectonic failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

$builtPdf = Join-Path $sourceDir "main.pdf"
if (-not (Test-Path -LiteralPath $builtPdf)) {
    throw "Expected PDF was not produced: $builtPdf"
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
Copy-Item -LiteralPath $builtPdf -Destination $outputPdf -Force

$pdfInfo = Get-Command pdfinfo -ErrorAction SilentlyContinue
if ($pdfInfo) {
    $pageLine = & $pdfInfo.Source $outputPdf | Select-String "^Pages:"
    if (-not $pageLine) {
        throw "pdfinfo could not verify the generated PDF"
    }
    Write-Output $pageLine.Line
}

$hash = (Get-FileHash -LiteralPath $outputPdf -Algorithm SHA256).Hash
Write-Output "PDF: $outputPdf"
Write-Output "SHA256: $hash"
