[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

function Find-CodexToolchainRoot {
	$currentDirectory = Get-Item -LiteralPath $PSScriptRoot
	while ($null -ne $currentDirectory) {
		if ($currentDirectory.Name -ieq "Codex") {
			$repositoryRelativeToolchains = Join-Path $currentDirectory.FullName "Toolchains"
			if (Test-Path -LiteralPath $repositoryRelativeToolchains -PathType Container) {
				return $repositoryRelativeToolchains
			}
		}
		$currentDirectory = $currentDirectory.Parent
	}

	$candidates = @()
	if (-not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
		$candidates += Join-Path $env:USERPROFILE "Documents\Codex\Toolchains"
	}
	$documentsPath = [Environment]::GetFolderPath("MyDocuments")
	if (-not [string]::IsNullOrWhiteSpace($documentsPath)) {
		$candidates += Join-Path $documentsPath "Codex\Toolchains"
	}

	foreach ($candidate in $candidates) {
		if (Test-Path -LiteralPath $candidate -PathType Container) {
			return $candidate
		}
	}

	if ($candidates.Count -gt 0) {
		return $candidates[0]
	}
	throw "Could not determine the Codex toolchain directory."
}

function Set-ToolchainEnvironment {
	$toolchainRoot = Find-CodexToolchainRoot

	if ([string]::IsNullOrWhiteSpace($env:VULKAN_SDK)) {
		$env:VULKAN_SDK = Join-Path $toolchainRoot "VulkanSDK\1.4.350.0"
	}
	if ([string]::IsNullOrWhiteSpace($env:DLSS_SDK)) {
		$env:DLSS_SDK = Join-Path $toolchainRoot "DLSS"
	}

	if (-not (Test-Path -LiteralPath $env:VULKAN_SDK -PathType Container)) {
		throw "Vulkan SDK was not found at $env:VULKAN_SDK. Run .\checkEnvironment.ps1."
	}
	if (-not (Test-Path -LiteralPath $env:DLSS_SDK -PathType Container)) {
		throw "DLSS SDK was not found at $env:DLSS_SDK. Run .\checkEnvironment.ps1."
	}
}

function Repair-DuplicatePathEnvironment {
	# Codex and a few other launchers can preserve both Path and PATH in the
	# Windows process environment. MSBuild's .NET Framework task host treats
	# those case-insensitive names as duplicate Hashtable keys and cannot
	# launch cl.exe. Keep the canonical Path entry when both are present.
	$pathKeys = @(
		[Environment]::GetEnvironmentVariables("Process").Keys |
			ForEach-Object { [string]$_ } |
			Where-Object { $_ -ieq "Path" }
	)
	if ($pathKeys.Count -gt 1 -and $pathKeys -contains "Path" -and $pathKeys -contains "PATH") {
		[Environment]::SetEnvironmentVariable("PATH", $null, "Process")
	}
}

function Find-CMake {
	$cmakeCommand = Get-Command cmake -ErrorAction SilentlyContinue
	if ($null -ne $cmakeCommand) {
		return $cmakeCommand.Source
	}

	$vsWherePath = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
	if (Test-Path -LiteralPath $vsWherePath -PathType Leaf) {
		$visualStudioPath = (& $vsWherePath -latest -products * `
			-requires Microsoft.VisualStudio.Component.VC.CMake.Project `
			-property installationPath | Select-Object -First 1)
		if (-not [string]::IsNullOrWhiteSpace($visualStudioPath)) {
			$bundledCmake = Join-Path $visualStudioPath.Trim() `
				"Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
			if (Test-Path -LiteralPath $bundledCmake -PathType Leaf) {
				return $bundledCmake
			}
		}
	}

	throw "CMake was not found. Install Visual Studio 2022 Build Tools with C++ CMake tools."
}

function Invoke-Checked {
	param(
		[string]$Executable,
		[string[]]$Arguments
	)

	& $Executable @Arguments
	if ($LASTEXITCODE -ne 0) {
		throw "$Executable failed with exit code $LASTEXITCODE."
	}
}

Set-ToolchainEnvironment
Repair-DuplicatePathEnvironment
$cmakePath = Find-CMake
$nativeSource = Join-Path $PSScriptRoot "native\ngx_shim"
$nativeBuild = Join-Path $PSScriptRoot "build\cmake\ngx_shim\release"

Write-Host "Configuring the NVIDIA NGX shim..." -ForegroundColor Cyan
Invoke-Checked $cmakePath @(
	"-S", $nativeSource,
	"-B", $nativeBuild,
	"-G", "Visual Studio 17 2022",
	"-A", "x64"
)

Write-Host "Building the NVIDIA NGX shim..." -ForegroundColor Cyan
Invoke-Checked $cmakePath @(
	"--build", $nativeBuild,
	"--config", "Release"
)

$shimPath = Join-Path $PSScriptRoot "build\native\ngx_shim\release\ngxshim.dll"
if (-not (Test-Path -LiteralPath $shimPath -PathType Leaf)) {
	throw "The native build completed but $shimPath was not produced."
}

Write-Host "Native NGX shim is ready: $shimPath" -ForegroundColor Green
