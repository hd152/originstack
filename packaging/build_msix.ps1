<#
.SYNOPSIS
    Builds an MSIX package of the Windows desktop app, for the Microsoft Store or for local testing.

.DESCRIPTION
    Wraps an existing PyInstaller onedir build (packaging\dist\OriginStack\, from
    build_windows.ps1, or the contents of a release OriginStack-<version>-windows-x64.zip given
    with -DistDir) into packaging\dist\OriginStack-<VERSION>-x64.msix with makeappx.exe from the
    Windows SDK. Nothing is added to the app itself.

    Store submission: reserve the app name in Partner Center, then pass the Package/Identity/Name
    and Package/Identity/Publisher values it shows (Product management > Product identity) as
    -IdentityName and -Publisher. The package is uploaded unsigned; the Store signs it.

    Local test install: -TestSign creates (or reuses) a self-signed certificate whose subject is
    -Publisher, signs the package with it and exports packaging\dist\OriginStack-test.cer. Trust it
    once from an elevated prompt, then double-click the .msix:
        Import-Certificate -FilePath OriginStack-test.cer -CertStoreLocation Cert:\LocalMachine\TrustedPeople
    A test-signed package is not for distribution.

    The package version is the repo's VERSION with a fourth part of 0 (the Store reserves the
    fourth part and rejects a non-zero one).
#>
param(
    [string]$DistDir = '',
    [string]$IdentityName = 'OriginStack',
    [string]$Publisher = 'CN=OriginStack',
    [string]$PublisherDisplayName = 'OriginStack',
    [switch]$TestSign,
    [string]$SdkBin = ''
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
if (-not $DistDir) { $DistDir = "$PSScriptRoot\dist\OriginStack" }
if (-not (Test-Path "$DistDir\OriginStack.exe")) {
    throw "$DistDir\OriginStack.exe not found -- run packaging\build_windows.ps1 first or pass -DistDir"
}

# Windows SDK tools (makeappx, signtool): newest x64 folder that has makeappx.exe.
if (-not $SdkBin) {
    $bins = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\10.*\x64" -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path "$($_.FullName)\makeappx.exe" } |
        Sort-Object { [version](Split-Path (Split-Path $_.FullName -Parent) -Leaf) } -Descending
    if (-not $bins) { throw "makeappx.exe not found. Install the Windows 10/11 SDK (winget install Microsoft.WindowsSDK.10.0.26100)" }
    $SdkBin = @($bins)[0].FullName
}
$makeappx = "$SdkBin\makeappx.exe"
$signtool = "$SdkBin\signtool.exe"

$version = (Get-Content "$Root\VERSION" -Raw).Trim()
if ($version -notmatch '^\d+\.\d+\.\d+$') { throw "VERSION '$version' is not MAJOR.MINOR.PATCH" }
$msixVersion = "$version.0"

# Layout: the app folder plus the manifest and the tile images, in a clean staging folder.
$stage = "$PSScriptRoot\build\msix"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path "$stage\Assets" | Out-Null
Copy-Item "$DistDir\*" $stage -Recurse -Force

# Tile images: pre-rendered from icon.ico into packaging\msix\Assets (re-render with Pillow if the icon changes).
Copy-Item "$PSScriptRoot\msix\Assets\*.png" "$stage\Assets" -Force

$manifest = @"
<?xml version="1.0" encoding="utf-8"?>
<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
         xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
         xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"
         IgnorableNamespaces="uap rescap">
  <Identity Name="$IdentityName" Publisher="$Publisher" Version="$msixVersion" ProcessorArchitecture="x64" />
  <Properties>
    <DisplayName>OriginStack</DisplayName>
    <PublisherDisplayName>$PublisherDisplayName</PublisherDisplayName>
    <Logo>Assets\StoreLogo.png</Logo>
  </Properties>
  <Dependencies>
    <TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.17763.0" MaxVersionTested="10.0.26100.0" />
  </Dependencies>
  <Resources>
    <Resource Language="en-us" />
  </Resources>
  <Applications>
    <Application Id="OriginStack" Executable="OriginStack.exe" EntryPoint="Windows.FullTrustApplication">
      <uap:VisualElements DisplayName="OriginStack"
                          Description="Stack, calibrate and finish astrophotography images, free and open source."
                          BackgroundColor="transparent"
                          Square150x150Logo="Assets\Square150x150Logo.png"
                          Square44x44Logo="Assets\Square44x44Logo.png">
        <uap:DefaultTile Wide310x150Logo="Assets\Wide310x150Logo.png" Square310x310Logo="Assets\Square310x310Logo.png" />
      </uap:VisualElements>
    </Application>
  </Applications>
  <Capabilities>
    <rescap:Capability Name="runFullTrust" />
  </Capabilities>
</Package>
"@
[System.IO.File]::WriteAllText("$stage\AppxManifest.xml", $manifest, (New-Object System.Text.UTF8Encoding($false)))

$out = "$PSScriptRoot\dist\OriginStack-$version-x64.msix"
New-Item -ItemType Directory -Force "$PSScriptRoot\dist" | Out-Null
if (Test-Path $out) { Remove-Item $out -Force }
Write-Host "Packing $out (identity $IdentityName, $Publisher, $msixVersion)"
& $makeappx pack /d $stage /p $out /o
if ($LASTEXITCODE -ne 0) { throw "makeappx failed (exit $LASTEXITCODE)" }

if ($TestSign) {
    $cert = Get-ChildItem Cert:\CurrentUser\My | Where-Object { $_.Subject -eq $Publisher -and $_.FriendlyName -eq 'OriginStack test' } | Select-Object -First 1
    if (-not $cert) {
        $cert = New-SelfSignedCertificate -Type Custom -Subject $Publisher -KeyUsage DigitalSignature `
            -FriendlyName 'OriginStack test' -CertStoreLocation Cert:\CurrentUser\My `
            -TextExtension @('2.5.29.37={text}1.3.6.1.5.5.7.3.3', '2.5.29.19={text}')
    }
    & $signtool sign /fd SHA256 /sha1 $cert.Thumbprint $out
    if ($LASTEXITCODE -ne 0) { throw "signtool failed (exit $LASTEXITCODE)" }
    Export-Certificate -Cert $cert -FilePath "$PSScriptRoot\dist\OriginStack-test.cer" | Out-Null
    Write-Host "Test-signed. Trust packaging\dist\OriginStack-test.cer (see -? help), then open the .msix."
}
Write-Host "Done: $out"
