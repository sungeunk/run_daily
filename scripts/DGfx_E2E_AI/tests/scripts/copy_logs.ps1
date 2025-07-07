Param
(
    [Parameter(Mandatory=$True)]
    [string]$Source
)

Write-Host "$(Get-Date) - Scanning for result directory..."
$ResultDirectory = Get-ChildItem $Source | Where-Object { $_.PSIsContainer -and $_.Name.StartsWith('Model_Perf_Results')}
$ResultDirectory.FullName

#Get Benchmark summary logs
$SummaryResultFiles = Get-ChildItem -Path $ResultDirectory.FullName -Recurse -Filter "Bench_output*"
ForEach ($File in $($SummaryResultFiles.FullName))
{
    Write-Host "$(Get-Date) - Copying $File file..."
    Copy-Item $File $PSScriptRoot\..\logs -Recurse -Force
}

#Get Benchmark details logs
ForEach ($FullPath in $($ResultDirectory.FullName))
{	
	Write-Host "$(Get-Date) - Scanning for log file..."
    if (Test-Path -Path $FullPath)
    {
        $LogFolder = Join-Path $FullPath "Perf_Test_Logs"
        $ResultLogFile = Get-ChildItem -Path $LogFolder
        
        ForEach ($FullLogFolderPath in $LogFolder) {
            if (Test-Path -Path $FullLogFolderPath)
            {
                Write-Host "$(Get-Date) - Copying $FullLogFolderPath directory..."
                Copy-Item $FullLogFolderPath $PSScriptRoot\..\logs -Recurse -Force
            }
        }
    }
    Write-Host "$(Get-Date) - Copied file completed."     
}

Write-Host "$(Get-Date) - Cleaning up old logs folder $ResultDirectory"
$SourceFolders = Get-ChildItem $Source

ForEach ($DirectoryPath in $SourceFolders) {
    Write-Host "$(Get-Date) Found directory '$DirectoryPath'. Removing..."
    Remove-Item -Path $DirectoryPath.FullName -Recurse
}
