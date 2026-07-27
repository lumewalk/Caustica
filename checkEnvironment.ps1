[CmdletBinding()]
param(
	[switch]$IncludeCacheSizes
)

$ErrorActionPreference = "Stop"

$script:ErrorCount = 0
$script:WarningCount = 0

function Write-Check {
	param(
		[ValidateSet("OK", "WARN", "ERROR", "INFO")]
		[string]$Level,
		[string]$Name,
		[string]$Details
	)

	switch ($Level) {
		"OK" {
			$color = "Green"
		}
		"WARN" {
			$color = "Yellow"
			$script:WarningCount++
		}
		"ERROR" {
			$color = "Red"
			$script:ErrorCount++
		}
		default {
			$color = "Cyan"
		}
	}

	Write-Host ("[{0,-5}] {1}: {2}" -f $Level, $Name, $Details) -ForegroundColor $color
}

function Get-ConfiguredEnvironmentVariable {
	param(
		[string]$Name,
		[string]$DefaultPath
	)

	$processValue = [Environment]::GetEnvironmentVariable($Name, "Process")
	if (-not [string]::IsNullOrWhiteSpace($processValue)) {
		return [pscustomobject]@{
			Value = $processValue
			Scope = "current process"
		}
	}

	foreach ($scope in @("User", "Machine")) {
		$value = [Environment]::GetEnvironmentVariable($Name, $scope)
		if (-not [string]::IsNullOrWhiteSpace($value)) {
			return [pscustomobject]@{
				Value = $value
				Scope = "$scope environment (restart the terminal to activate it)"
			}
		}
	}

	if (-not [string]::IsNullOrWhiteSpace($DefaultPath) -and
		(Test-Path -LiteralPath $DefaultPath -PathType Container)) {
		return [pscustomobject]@{
			Value = $DefaultPath
			Scope = "Codex toolchain default"
		}
	}

	return $null
}

function Find-Tool {
	param(
		[string]$Name,
		[string[]]$PreferredDirectories = @()
	)

	foreach ($directory in $PreferredDirectories) {
		if ([string]::IsNullOrWhiteSpace($directory)) {
			continue
		}

		$candidate = Join-Path $directory "$Name.exe"
		if (Test-Path -LiteralPath $candidate -PathType Leaf) {
			return (Get-Item -LiteralPath $candidate).FullName
		}
	}

	$command = Get-Command $Name -ErrorAction SilentlyContinue
	if ($null -ne $command) {
		return $command.Source
	}

	return $null
}

function Format-ByteSize {
	param([long]$Bytes)

	if ($Bytes -ge 1TB) {
		return "{0:N1} TB" -f ($Bytes / 1TB)
	}
	if ($Bytes -ge 1GB) {
		return "{0:N1} GB" -f ($Bytes / 1GB)
	}
	if ($Bytes -ge 1MB) {
		return "{0:N1} MB" -f ($Bytes / 1MB)
	}
	return "{0:N0} KB" -f ($Bytes / 1KB)
}

function Get-DirectoryByteSize {
	param([string]$Path)

	if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
		return 0
	}

	$measurement = Get-ChildItem -LiteralPath $Path -File -Recurse -Force -ErrorAction SilentlyContinue |
		Measure-Object -Property Length -Sum
	if ($null -eq $measurement.Sum) {
		return 0
	}
	return [long]$measurement.Sum
}

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
	return $null
}

Write-Host ""
Write-Host "Caustica development environment" -ForegroundColor White
Write-Host "Repository: $PSScriptRoot"
Write-Host ""

if ([Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT) {
	Write-Check "OK" "Operating system" ([Environment]::OSVersion.VersionString)
} else {
	Write-Check "ERROR" "Operating system" "This checker currently supports Windows only."
}

try {
	$driveRoot = (Get-Item -LiteralPath $PSScriptRoot).PSDrive.Root
	$drive = [System.IO.DriveInfo]::new($driveRoot)
	$freeSpace = $drive.AvailableFreeSpace
	$freeLevel = if ($freeSpace -lt 20GB) { "ERROR" } elseif ($freeSpace -lt 30GB) { "WARN" } else { "OK" }
	Write-Check $freeLevel "Free disk space" ("{0} on {1}" -f (Format-ByteSize $freeSpace), $drive.Name)
} catch {
	Write-Check "WARN" "Free disk space" "Could not read: $($_.Exception.Message)"
}

try {
	$computer = Get-CimInstance Win32_ComputerSystem -ErrorAction Stop
	Write-Check "INFO" "System memory" (Format-ByteSize ([long]$computer.TotalPhysicalMemory))
} catch {
	Write-Check "WARN" "System memory" "Could not query WMI/CIM."
}

try {
	$gpuNames = @(Get-CimInstance Win32_VideoController -ErrorAction Stop |
		ForEach-Object { $_.Name } |
		Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
	if ($gpuNames.Count -gt 0) {
		Write-Check "INFO" "Graphics adapter" ($gpuNames -join "; ")
	} else {
		Write-Check "WARN" "Graphics adapter" "No adapter was reported by WMI/CIM."
	}
} catch {
	Write-Check "WARN" "Graphics adapter" "Could not query WMI/CIM."
}

Write-Host ""
Write-Host "Required build tools" -ForegroundColor White

$toolchainRoot = Find-CodexToolchainRoot
$defaultVulkanSdk = if ($null -ne $toolchainRoot) {
	Join-Path $toolchainRoot "VulkanSDK\1.4.350.0"
} else {
	$null
}
$defaultDlssSdk = if ($null -ne $toolchainRoot) {
	Join-Path $toolchainRoot "DLSS"
} else {
	$null
}

$javaPath = Find-Tool "java"
if ($null -eq $javaPath) {
	Write-Check "ERROR" "Java" "JDK 25 was not found on PATH."
} else {
	$javaVersionText = (& $javaPath --version 2>&1 | Select-Object -First 1).ToString()
	if ($javaVersionText -match '(?:java|openjdk)\s+(\d+)') {
		$javaMajor = [int]$matches[1]
		if ($javaMajor -eq 25) {
			Write-Check "OK" "Java" "$javaVersionText ($javaPath)"
		} else {
			Write-Check "ERROR" "Java" "JDK 25 is required; found $javaVersionText ($javaPath)"
		}
	} else {
		Write-Check "WARN" "Java" "Could not parse version: $javaVersionText ($javaPath)"
	}
}

$gitPath = Find-Tool "git"
if ($null -eq $gitPath) {
	Write-Check "ERROR" "git" "Not found on PATH."
} else {
	Write-Check "OK" "git" $gitPath
}

$vsWherePath = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$msvcPath = Find-Tool "cl"
$visualStudioPath = $null
if (Test-Path -LiteralPath $vsWherePath -PathType Leaf) {
	$visualStudioPath = (& $vsWherePath -latest -products * `
		-requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
		-property installationPath | Select-Object -First 1)
	if (-not [string]::IsNullOrWhiteSpace($visualStudioPath)) {
		$visualStudioPath = $visualStudioPath.Trim()
	}
}

if ($null -ne $msvcPath) {
	Write-Check "OK" "C++ compiler" $msvcPath
} elseif (-not [string]::IsNullOrWhiteSpace($visualStudioPath)) {
	Write-Check "OK" "C++ compiler" "Visual Studio C++ tools at $visualStudioPath"
} else {
	Write-Check "ERROR" "C++ compiler" "Install Visual Studio 2022 Build Tools with Desktop development with C++."
}

$cmakeDirectories = @()
if (-not [string]::IsNullOrWhiteSpace($visualStudioPath)) {
	$cmakeDirectories += Join-Path $visualStudioPath "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
}
$cmakePath = Find-Tool "cmake" $cmakeDirectories
if ($null -eq $cmakePath) {
	Write-Check "ERROR" "cmake" "Not found on PATH or in Visual Studio Build Tools."
} else {
	Write-Check "OK" "cmake" $cmakePath
}

$vulkanConfig = Get-ConfiguredEnvironmentVariable "VULKAN_SDK" $defaultVulkanSdk
$vulkanDirectories = @()
if ($null -eq $vulkanConfig) {
	Write-Check "ERROR" "VULKAN_SDK" "Not configured. Install the LunarG Vulkan SDK."
} elseif (-not (Test-Path -LiteralPath $vulkanConfig.Value -PathType Container)) {
	Write-Check "ERROR" "VULKAN_SDK" "Configured path does not exist: $($vulkanConfig.Value)"
} else {
	Write-Check "OK" "VULKAN_SDK" "$($vulkanConfig.Value) [$($vulkanConfig.Scope)]"
	$vulkanDirectories = @(
		(Join-Path $vulkanConfig.Value "Bin"),
		(Join-Path $vulkanConfig.Value "bin")
	)
	$vulkanHeader = Join-Path $vulkanConfig.Value "Include\vulkan\vulkan.h"
	if (-not (Test-Path -LiteralPath $vulkanHeader -PathType Leaf)) {
		$vulkanHeader = Join-Path $vulkanConfig.Value "include\vulkan\vulkan.h"
	}
	if (Test-Path -LiteralPath $vulkanHeader -PathType Leaf) {
		Write-Check "OK" "Vulkan headers" $vulkanHeader
	} else {
		Write-Check "ERROR" "Vulkan headers" "vulkan.h was not found under $($vulkanConfig.Value)"
	}
}

foreach ($shaderTool in @("slangc", "glslangValidator", "spirv-val")) {
	$shaderToolPath = Find-Tool $shaderTool $vulkanDirectories
	if ($null -eq $shaderToolPath) {
		Write-Check "ERROR" $shaderTool "Not found in VULKAN_SDK or on PATH."
	} else {
		Write-Check "OK" $shaderTool $shaderToolPath
	}
}

$dlssConfig = Get-ConfiguredEnvironmentVariable "DLSS_SDK" $defaultDlssSdk
if ($null -eq $dlssConfig) {
	Write-Check "ERROR" "DLSS_SDK" "Not configured. Download and extract the NVIDIA DLSS SDK."
} elseif (-not (Test-Path -LiteralPath $dlssConfig.Value -PathType Container)) {
	Write-Check "ERROR" "DLSS_SDK" "Configured path does not exist: $($dlssConfig.Value)"
} else {
	Write-Check "OK" "DLSS_SDK" "$($dlssConfig.Value) [$($dlssConfig.Scope)]"
	$requiredDlssFiles = @(
		"include\nvsdk_ngx.h",
		"lib\Windows_x86_64\x64\nvsdk_ngx_d.lib",
		"lib\Windows_x86_64\rel\nvngx_dlssd.dll",
		"lib\Windows_x86_64\rel\nvngx_dlssg.dll"
	)
	foreach ($relativePath in $requiredDlssFiles) {
		$fullPath = Join-Path $dlssConfig.Value $relativePath
		if (Test-Path -LiteralPath $fullPath -PathType Leaf) {
			Write-Check "OK" "DLSS component" $relativePath
		} else {
			Write-Check "ERROR" "DLSS component" "Missing $relativePath"
		}
	}
}

if ($IncludeCacheSizes) {
	Write-Host ""
	Write-Host "Disposable build and cache data" -ForegroundColor White

	$cachePaths = [ordered]@{
		"Project build" = (Join-Path $PSScriptRoot "build")
		"Minecraft run data" = (Join-Path $PSScriptRoot "run")
		"Project Gradle cache" = (Join-Path $PSScriptRoot ".gradle")
	}
	if (-not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
		$cachePaths["User Gradle cache"] = Join-Path $env:USERPROFILE ".gradle\caches"
	}

	foreach ($entry in $cachePaths.GetEnumerator()) {
		$size = Get-DirectoryByteSize $entry.Value
		Write-Check "INFO" $entry.Key ("{0} ({1})" -f (Format-ByteSize $size), $entry.Value)
	}
}

Write-Host ""
if ($script:ErrorCount -eq 0) {
	Write-Host ("Environment is ready. Warnings: {0}." -f $script:WarningCount) -ForegroundColor Green
	exit 0
}

Write-Host ("Environment is not ready: {0} error(s), {1} warning(s)." -f `
	$script:ErrorCount, $script:WarningCount) -ForegroundColor Red
Write-Host "See docs/developer_guide.md for setup instructions."
exit 1
