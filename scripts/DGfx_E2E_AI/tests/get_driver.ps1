# Get non-Intel video controller information
$gpus = Get-WmiObject Win32_VideoController | Where-Object { $_.Status -eq "OK" } | Select-Object AdapterCompatibility, DriverVersion, VideoProcessor

$output = @()

foreach ($gpu in $gpus) {
    # Extract the required information
    $vendor = $gpu.AdapterCompatibility
    $driver = $gpu.DriverVersion
    $product = $gpu.VideoProcessor

    # Create a custom object with the required format
    $result = [PSCustomObject]@{
        Vendor = $vendor
        Driver = $driver
        Product = $product
    }

    # Add the result to the output array
    $output += $result
}

# Write output to a text file in the specified format
[System.IO.File]::WriteAllText("driverInfo.txt", ($output | ForEach-Object { "Vendor: $($_.Vendor)`nDriver: $($_.Driver)`nProduct: $($_.Product)`n" }))
[System.IO.File]::WriteAllText("driverInfo.json", ($output | ConvertTo-Json))