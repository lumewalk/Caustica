param(
	[switch]$FrameStats,
	[switch]$View20FullAudit
)

$ErrorActionPreference = "Stop"

$causticaRoot = $PSScriptRoot

Push-Location $causticaRoot
try {
	& (Join-Path $causticaRoot "buildNative.ps1")
	$javaToolOptions = '-Xmx8G -XX:+UseCompactObjectHeaders -XX:+AlwaysPreTouch -XX:+UseStringDeduplication -XX:+UseZGC'
	if ($FrameStats) {
		Write-Host "Caustica frame statistics are enabled."
	}
	$javaToolOptions += " -Dcaustica.rt.frameStats=$($FrameStats.IsPresent.ToString().ToLowerInvariant())"
	if ($View20FullAudit) {
		Write-Host "Caustica debug view 20 full audit is enabled."
		$javaToolOptions += " -Dcaustica.rt.view20FullAudit=true"
	}
	$env:JAVA_TOOL_OPTIONS = $javaToolOptions
	.\gradlew.bat --no-daemon runClient --args="--renderDebugLabels --graphicsBackend VULKAN"
	if ($LASTEXITCODE -ne 0) {
		throw "Gradle runClient failed with exit code $LASTEXITCODE."
	}
} finally {
	Pop-Location
}
