import xml.etree.ElementTree as ET
import sys

from utils_aiml_intel.metrics import BenchmarkRecord
from utils_aiml_intel.setup_logging import (get_gtax_test_dir,os)

test_path = get_gtax_test_dir()
log_path = test_path / "logs"
clpeak_exe_path = "clpeak\\clpeak.exe"

clpeak_devices = os.popen(f"{clpeak_exe_path} -d 1").read().splitlines()
temp = filter(lambda y: y != '', clpeak_devices)
clpeak_devices = list(temp)

if len(clpeak_devices) == 0: 
  print("No available devices were found - check drivers and check temp\\clpeak\\build")
  exit(1)

if len(sys.argv) <= 1:
  os.system(f"{clpeak_exe_path} -p 0 -f {log_path}\\clpeak_result.xml")
else:
  if int(sys.argv[1]) < len(clpeak_devices) and int(sys.argv[1]) >= 0:
    os.system(f"{clpeak_exe_path} -p {int(sys.argv[1])} -f {log_path}\\clpeak_result.xml")
  else:
    print("The input option is outside the range of available devices.")
    exit(1)

file_name = "clpeak_execution_results"

result_xml = log_path / "clpeak_result.xml"

xml_tree = ET.parse(result_xml)
xml_root = xml_tree.getroot()
xml_data = xml_root[0][0] # offset for this xml file (output structure is stupid)

# need to convert Element Array to a regular string array to check if metric is present and what index. 
data_index =[]
for x in  xml_data:
  data_index.append(x.tag)

metric_entries = []
record = BenchmarkRecord("clpeak","NA", xml_root[0].get("name"),xml_root[0][0].get('name'), "NA", "NA")
record.config.customized["Application Version"] = os.popen(f"{clpeak_exe_path} -v").read()[16:-1]
record.config.customized["Driver"] = xml_root[0][0].get('driver_version')
record.config.customized["Compute Units"] = xml_root[0][0].get('compute_units')
record.config.customized["Clock Frequency (MHz)"] = xml_root[0][0].get('clock_frequency')

for x in data_index:
  metric_index: int = data_index.index(x)
  child_metric_size = len(list(xml_data[metric_index]))
  if child_metric_size > 0:
    for y in range(child_metric_size): 
      record.metrics.customized[f"{xml_data[metric_index].tag.replace('_', ' ').title()} ({xml_data[metric_index].get('unit').upper()}), {xml_data[metric_index][y].tag}"] = xml_data[metric_index][y].text
  else:
    record.metrics.customized[f"{xml_data[metric_index].tag.replace('_', ' ').title()} ({xml_data[metric_index].get('unit').upper()})"] = xml_data[metric_index].text

metric_entries.append(record)

BenchmarkRecord.save_as_csv(log_path / file_name,metric_entries)
BenchmarkRecord.save_as_json(log_path / file_name,metric_entries)
BenchmarkRecord.save_as_txt(log_path / file_name,metric_entries)
