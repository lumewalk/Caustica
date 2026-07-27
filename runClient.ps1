$ErrorActionPreference = "Stop"

$causticaRoot = $PSScriptRoot

Push-Location $causticaRoot
try {
	& (Join-Path $causticaRoot "buildNative.ps1")
	$env:JAVA_TOOL_OPTIONS='-Xmx8G -XX:+UseCompactObjectHeaders -XX:+AlwaysPreTouch -XX:+UseStringDeduplication -XX:+UseZGC'
	.\gradlew.bat --no-daemon runClient --args="--renderDebugLabels --graphicsBackend VULKAN"
	if ($LASTEXITCODE -ne 0) {
		throw "Gradle runClient failed with exit code $LASTEXITCODE."
	}
} finally {
	Pop-Location
}
