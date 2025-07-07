# Get System Information Script

# Get OS Information
$osInfo = Get-CimInstance Win32_OperatingSystem
$osName = $osInfo.Caption
$osVersion = $osInfo.Version

# Get Processor Information
$processor = Get-CimInstance Win32_Processor
$processorName = $processor.Name

# Get BaseBoard Information
$baseBoard = Get-CimInstance Win32_BaseBoard
$baseBoardProduct = $baseBoard.Product

# Get Physical Memory Information
$physicalMemory = Get-CimInstance Win32_PhysicalMemory
$totalMemory = ($physicalMemory | Measure-Object -Property Capacity -Sum)
$totalMemoryGB = [math]::Round($totalMemory.Sum / 1GB, 2)
$memorySpeed = $physicalMemory[0].Speed

# Create system info object
$systemInfo = @{
    OSName = $osName
    OSVersion = $osVersion
    Processor = $processorName
    BaseBoardProduct = $baseBoardProduct
    InstalledPhysicalMemoryGB = $totalMemoryGB
    MemorySpeedMHz = $memorySpeed
}

# Display Information
Write-Host "System Information:" -ForegroundColor Green
Write-Host "=================="
Write-Host "OS Name: $osName"
Write-Host "OS Version: $osVersion"
Write-Host "Processor: $processorName"
Write-Host "BaseBoard Product: $baseBoardProduct"
Write-Host "Installed Physical Memory (GB): $totalMemoryGB"
Write-Host "Memory Speed (MHz): $memorySpeed"

# Save to TXT file
$txtContent = @"
System Information
==================
OS Name: $osName
OS Version: $osVersion
Processor: $processorName
BaseBoard Product: $baseBoardProduct
Installed Physical Memory (GB): $totalMemoryGB
Memory Speed (MHz): $memorySpeed
"@

$txtPath = Join-Path $PSScriptRoot "SystemInfo.txt"
[System.IO.File]::WriteAllText($txtPath, $txtContent)
Write-Host "`nSystem information saved to: $txtPath" -ForegroundColor Yellow

# Save to JSON file
$jsonPath = Join-Path $PSScriptRoot "SystemInfo.json"
[System.IO.File]::WriteAllText($jsonPath, ($systemInfo | ConvertTo-Json))
Write-Host "System information saved to: $jsonPath" -ForegroundColor Yellow