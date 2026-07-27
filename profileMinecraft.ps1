param(
	[int]$TargetPid = 0,
	[string]$RecordingName = "mc",
	[string]$Settings = "profile",
	[ValidateRange(0, 3600)]
	[int]$DurationSeconds = 0
)

$ErrorActionPreference = "Stop"

function Find-JdkTool {
	param(
		[Parameter(Mandatory = $true)]
		[string]$ToolName
	)

	$command = Get-Command $ToolName -ErrorAction SilentlyContinue
	if ($null -ne $command) {
		return $command.Source
	}

	$candidates = @()
	if ($env:JAVA_HOME) {
		$candidates += Join-Path $env:JAVA_HOME "bin\$ToolName.exe"
	}

	$javaCommand = Get-Command java -ErrorAction SilentlyContinue
	if ($null -ne $javaCommand) {
		$candidates += Join-Path (Split-Path -Parent $javaCommand.Source) "$ToolName.exe"
	}

	$jdkRoots = @(
		(Join-Path $env:ProgramFiles "Java"),
		(Join-Path $env:ProgramFiles "Eclipse Adoptium")
	)

	foreach ($jdkRoot in $jdkRoots) {
		if (-not (Test-Path -LiteralPath $jdkRoot)) {
			continue
		}

		$candidates += Get-ChildItem -LiteralPath $jdkRoot -Directory -ErrorAction SilentlyContinue |
			Sort-Object LastWriteTime -Descending |
			ForEach-Object { Join-Path $_.FullName "bin\$ToolName.exe" }
	}

	foreach ($candidate in $candidates) {
		if ($candidate -and (Test-Path -LiteralPath $candidate)) {
			return (Resolve-Path -LiteralPath $candidate).Path
		}
	}

	return $null
}

$jcmdPath = Find-JdkTool "jcmd"
if ($null -eq $jcmdPath) {
	throw "Could not find jcmd. Install a full JDK, put its bin directory on PATH, or set JAVA_HOME."
}

function Invoke-Jcmd {
	param(
		[Parameter(ValueFromRemainingArguments = $true)]
		[string[]]$JcmdArgs
	)

	& $jcmdPath @JcmdArgs
	if ($LASTEXITCODE -ne 0) {
		throw "jcmd failed: jcmd $($JcmdArgs -join ' ')"
	}
}

function Get-JcmdProcessList {
	$lines = & $jcmdPath
	if ($LASTEXITCODE -ne 0) {
		throw "jcmd failed while listing JVMs"
	}

	foreach ($line in $lines) {
		if ($line -match '^\s*(\d+)\s+(.+?)\s*$') {
			[pscustomobject]@{
				Id = [int]$matches[1]
				Command = $matches[2]
				Raw = $line
			}
		}
	}
}

$processes = @(Get-JcmdProcessList)

if ($TargetPid -ne 0) {
	$target = $processes | Where-Object { $_.Id -eq $TargetPid } | Select-Object -First 1
	if ($null -eq $target) {
		throw "Could not find JVM with PID $TargetPid."
	}
} else {
	$candidates = @(
		$processes | Where-Object {
			$_.Command -match 'net\.fabricmc\.devlaunchinjector\.Main|net\.minecraft\.client\.main\.Main|Minecraft' -and
			$_.Command -notmatch 'Gradle|GradleDaemon|GradleWrapperMain|jdk\.jcmd|JCmd|JMC'
		}
	)

	if ($candidates.Count -eq 0) {
		Write-Host "No Minecraft JVM found. Current jcmd output:"
		$processes | ForEach-Object { Write-Host "  $($_.Raw)" }
		throw "Start the Minecraft client first, or pass -TargetPid <pid>."
	}

	if ($candidates.Count -eq 1) {
		$target = $candidates[0]
	} else {
		Write-Host "Multiple Minecraft-like JVMs found:"
		for ($i = 0; $i -lt $candidates.Count; $i++) {
			Write-Host ("  [{0}] {1}" -f ($i + 1), $candidates[$i].Raw)
		}

		$selection = Read-Host "Select target [1]"
		if ([string]::IsNullOrWhiteSpace($selection)) {
			$selection = "1"
		}

		$index = [int]$selection - 1
		if ($index -lt 0 -or $index -ge $candidates.Count) {
			throw "Invalid selection: $selection"
		}

		$target = $candidates[$index]
	}
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outDir = Join-Path $PSScriptRoot "run/jfr"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$jfrPath = Join-Path $outDir "$RecordingName-$timestamp.jfr"
$targetPidText = [string]$target.Id
$started = $false

Write-Host "Target JVM: $($target.Raw)"
Write-Host "Recording:  $jfrPath"

try {
	Invoke-Jcmd $targetPidText "JFR.start" "name=$RecordingName" "settings=$Settings" "filename=$jfrPath"
	$started = $true
	Write-Host ""
	if ($DurationSeconds -gt 0) {
		Write-Host "Profiling for $DurationSeconds seconds..."
		Start-Sleep -Seconds $DurationSeconds
	} else {
		Read-Host "Profiling started. Press Enter to stop"
	}
} finally {
	if ($started) {
		Write-Host "Stopping JFR recording..."
		try {
			Invoke-Jcmd $targetPidText "JFR.stop" "name=$RecordingName" "filename=$jfrPath"
			Write-Host "Saved JFR recording to $jfrPath"
		} catch {
			Write-Warning $_.Exception.Message
			Write-Warning "The target JVM may have exited before the recording was stopped."
		}
	}
}
