$igfx = [System.Collections.Generic.Dictionary[string, string]]::new()
$igfx["13C0"] = "AMD iGFX"
$igfx["A780"] = "Intel iGFX"

$nv = [System.Collections.Generic.Dictionary[string, string]]::new()
$nv["2882"] = "4060"
$nv["2803"] = "4060TI"
$nv["2709"] = "4070"
$nv["2684"] = "4090"
$nv["25B0"] = "A1000"
$nv["2F04"] = "5070"
$nv["2D04"] = "5060TI"

$amd = [System.Collections.Generic.Dictionary[string, string]]::new()
$amd["150E"] = "880M/890M"

$acm = [System.Collections.Generic.Dictionary[string, string]]::new()
$acm["56A0"] = "A770"
$acm["56A1"] = "A750"
$acm["56A2"] = "A580"
$acm["5690"] = "A770M"
$acm["5691"] = "A730M"
$acm["5692"] = "A550M"
$acm["56C0"] = "170"
$acm["56A5"] = "A380"
$acm["56A6"] = "A310"
$acm["5693"] = "A370M"
$acm["5694"] = "A350M"
$acm["5695"] = "A200M"
$acm["56B0"] = "A30M"
$acm["56B1"] = "A40/A50"
$acm["56BA"] = "A380E"
$acm["56BB"] = "A310E"
$acm["56BC"] = "A370E"
$acm["56BD"] = "A350E"
$acm["56BE"] = "A750E"
$acm["56BF"] = "A580E"
$acm["56C1"] = "140"
$acm["5696"] = "A570M"
$acm["5697"] = "A530M"
$acm["56B2"] = "A60M"
$acm["56B3"] = "A60"
$acm["56C2"] = "170V"
$acm["56AF"] = "A760A"

$bmg = [System.Collections.Generic.Dictionary[string, string]]::new()
$bmg["E20B"] = "B580"
$bmg["E20C"] = "B570"
$bmg["E210"] = "B35"
$bmg["E212"] = "B93"
$bmg["E215"] = "NEX1"
$bmg["E216"] = "NEX2"
$bmg["E221"] = "G31_DT2"
$bmg["E220"] = "G31_DT1+"
$bmg["E222"] = "G31_DT3-"
$bmg["E223"] = "G31_DT1_WKS"

$mtl = [System.Collections.Generic.Dictionary[string, string]]::new()
$mtl["7D55"] = "H682"
$mtl["7DD5"] = "H682"
$mtl["7D45"] = "U281"

$lnl = [System.Collections.Generic.Dictionary[string, string]]::new()
$lnl["64A0"] = "130V/140V"
$lnl["6420"] = "130V/140V"

$arl = [System.Collections.Generic.Dictionary[string, string]]::new()
$arl["7D51"] = "130T/140T"
$arl["7DD1"] = "H682"
$arl["7D45"] = "7D41"
$arl["7D41"] = "U281"

$ptl = [System.Collections.Generic.Dictionary[string, string]]::new()
$ptl["B080"] = "H 12Xe"
$ptl["B081"] = "H 12Xe"
$ptl["B082"] = "H 12Xe"
$ptl["B083"] = "H 12Xe"
$ptl["B08F"] = "H 12Xe"
$ptl["B090"] = "U 4Xe"
$ptl["B0A0"] = "H 4Xe"
$ptl["B0B0"] = "H 4Xe (D2D)"

# $acm = @('4905', '56A0', '56A1', '56A2', '5690', '5691', '5692', '56C0', '56A5', '56A6', '5693', '5694', '5695', '56B0', '56B1', '56BA', '56BB', '56BC', '56BD', '56BE', '56BF', '56C1', '5696', '5697', '56B2', '56B3', '56C2', '56AF')
# $bmg = @('E201', 'E202', 'E204', 'E205', 'E208', 'E20B', 'E20C', 'E20D', 'E20E', 'E20F', 'E212', 'E220', 'E221', 'E222', 'E210')
# $mtl = @('7D40', '7D55', '7DD5', '7D45', '7D60')
# $lnl = @('64A0', '6420', '64B0')
# $arl = @('7D51', '7DD1', '7D41')
# $ptl = @('B080', 'B081', 'B082', 'B083', 'B08F', 'B090', 'B0A0', 'B0FF', 'B0B0')

# Function to save deviceID output
function Create-ResultDeviceID {
    param (
        [string]$platform,
        [string]$deviceId,
        [string]$product = "N/A"
    )
    return [PSCustomObject]@{
        Platform = $platform
        DeviceID = $deviceId
        Product = $product
    }
}

$gfxDevices = Get-WmiObject win32_VideoController | Where-Object { $_.Status -eq "OK" } | Where-Object {
    # Filter out iGFX devices
    if ($_.PNPDeviceID -like "*VEN_8086*") {
        if ($_.PNPDeviceID -match '.*VEN_8086\&DEV_([0-9a-fA-F]+)\&SUBSYS_[0-9a-fA-F]{4}([0-9a-fA-F]{4}).*') {
            $deviceId = $Matches[1]
            return !$igfx.ContainsKey($deviceId)
        }
    } elseif ($_.PNPDeviceID -like "*VEN_1002*") {
        if ($_.PNPDeviceID -match '.*VEN_1002\&DEV_([0-9a-fA-F]+)\&SUBSYS_[0-9a-fA-F]{4}([0-9a-fA-F]{4}).*') {
            $deviceId = $Matches[1]
            return !$igfx.ContainsKey($deviceId)
        }
    }
    return $true
}

$output = @()

foreach($gfxDevice in $gfxDevices) 
{
    Write-Host "$(Get-Date) - $($gfxDevice.PNPDeviceID) $($gfxDevice.Name) for driver version $GfxDriverVersion - $($gfxDevice.Status)"
    if ($gfxDevice.PNPDeviceID -like "*VEN_8086*") {
        Write-Output "Intel Device Detected"
        if ($gfxDevice.PNPDeviceID -match '.*VEN_8086\&DEV_([0-9a-fA-F]+)\&SUBSYS_[0-9a-fA-F]{4}([0-9a-fA-F]{4}).*') {
            $deviceId = $Matches[1]
            if ($acm.ContainsKey($deviceId)) {
                Write-Output "ACM"
                $output += Create-ResultDeviceID -platform "ACM" -deviceId $deviceId -product $acm[$deviceId]
            } elseif ($bmg.ContainsKey($deviceId)) {
                Write-Output "BMG"
                $output += Create-ResultDeviceID -platform "BMG" -deviceId $deviceId -product $bmg[$deviceId]
            } elseif ($mtl.ContainsKey($deviceId)) {
                Write-Output "MTL"
                $output += Create-ResultDeviceID -platform "MTL" -deviceId $deviceId -product $mtl[$deviceId]
            } elseif ($lnl.ContainsKey($deviceId)) {
                Write-Output "LNL"
                $output += Create-ResultDeviceID -platform "LNL" -deviceId $deviceId -product $lnl[$deviceId]
            } elseif ($arl.ContainsKey($deviceId)) {
                Write-Output "ARL"
                $output += Create-ResultDeviceID -platform "ARL" -deviceId $deviceId -product $arl[$deviceId]
            } elseif ($ptl.ContainsKey($deviceId)) {
                Write-Output "PTL"
                $output += Create-ResultDeviceID -platform "PLT" -deviceId $deviceId -product $ptl[$deviceId]
            } else {
                Write-Output "Not Intel device, checking other vendors ..."
            }
        }
    } elseif ($gfxDevice.PNPDeviceID -like "*VEN_10DE*") {
        Write-Output "Nvidia Device Detected"
        if ($gfxDevice.PNPDeviceID -match '.*VEN_10DE\&DEV_([0-9a-fA-F]+)\&SUBSYS_[0-9a-fA-F]{4}([0-9a-fA-F]{4}).*') {
            $deviceId = $Matches[1]
            if ($nv.ContainsKey($deviceId)) {
                Write-Output "NV"
                $output += Create-ResultDeviceID -platform "NV" -deviceId $deviceId -product $nv[$deviceId]
                
            } else {
                Write-Output "Not Nvidia device, checking other vendors ..."
            }
        }
    } elseif ($gfxDevice.PNPDeviceID -like "*VEN_1002*") {
        Write-Output "AMD Device Detected"
        if ($gfxDevice.PNPDeviceID -match '.*VEN_1002\&DEV_([0-9a-fA-F]+)\&SUBSYS_[0-9a-fA-F]{4}([0-9a-fA-F]{4}).*') {
            $deviceId = $Matches[1]
            if ($amd.ContainsKey($deviceId)) {
                Write-Output "AMD"
                $output += Create-ResultDeviceID -platform "AMD" -deviceId $deviceId -product $amd[$deviceId]
            } else {
                Write-Output "Not AMD device, checking other vendors ..."
            }
        }
    }
}

if($output.Count -eq 0) {
    throw "Unknown Platform for Device ID: $deviceId. Please contact Russell Tan (russell.tan@intel.com) to add deviceID compatibility. As a stopgap solution to unblock this process, please add deviceID to get_playform.ps1 file."
} else {
    [System.IO.File]::WriteAllText("deviceID.txt", ($output | ForEach-Object { $_ | ConvertTo-Json -Compress }))
    [System.IO.File]::WriteAllText("deviceID.json", ($output | ConvertTo-Json))
}
