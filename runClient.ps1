$ErrorActionPreference = "Stop"

$causticaRoot = $PSScriptRoot

Push-Location $causticaRoot
try {
	& (Join-Path $causticaRoot "buildNative.ps1")
	.\gradlew.bat --stop
	$env:JAVA_TOOL_OPTIONS='-Xmx8G -XX:+UseCompactObjectHeaders -XX:+AlwaysPreTouch -XX:+UseStringDeduplication -XX:+UseZGC'
	.\gradlew.bat runClient --args="--renderDebugLabels --graphicsBackend VULKAN"
} finally {
	Pop-Location
}
