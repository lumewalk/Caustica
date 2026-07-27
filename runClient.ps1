param(
	[switch]$FrameStats
)

$ErrorActionPreference = "Stop"

$causticaRoot = $PSScriptRoot

Push-Location $causticaRoot
try {
	& (Join-Path $causticaRoot "buildNative.ps1")
	$javaToolOptions = '-Xmx8G -XX:+UseCompactObjectHeaders -XX:+AlwaysPreTouch -XX:+UseStringDeduplication -XX:+UseZGC'
	if ($FrameStats) {
		$javaToolOptions += ' -Dcaustica.rt.frameStats=true'
		Write-Host "Caustica frame statistics are enabled."
	}
	$env:JAVA_TOOL_OPTIONS = $javaToolOptions
	.\gradlew.bat --no-daemon runClient --args="--renderDebugLabels --graphicsBackend VULKAN"
	if ($LASTEXITCODE -ne 0) {
		throw "Gradle runClient failed with exit code $LASTEXITCODE."
	}
} finally {
	Pop-Location
}
