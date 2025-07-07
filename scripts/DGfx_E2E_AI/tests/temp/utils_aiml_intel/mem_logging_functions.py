import subprocess
import threading
import time


class MemLogger():
    
    def __init__(self, monitor=False):
        self.peak_gpu_memory_process_local = 0.0
        self.peak_gpu_memory_process_shared = 0.0
        self.peak_gpu_memory_process_dedicated = 0.0
        self.peak_gpu_memory_process_nonlocal = 0.0
        self.peak_gpu_memory_process_committed = 0.0
        self.peak_gpu_memory_engine_utilization_compute = 0.0
        self.peak_gpu_memory_engine_utilization_copy = 0.0

        self.gpu_memory_command_process_local = r'(((Get-Counter "\GPU Process Memory(*)\Local Usage").CounterSamples | where CookedValue).CookedValue | measure -sum).sum'
        self.gpu_memory_command_process_shared = r'(((Get-Counter "\GPU Process Memory(*)\Shared Usage").CounterSamples | where CookedValue).CookedValue | measure -sum).sum'
        self.gpu_memory_command_process_dedicated = r'(((Get-Counter "\GPU Process Memory(*)\Dedicated Usage").CounterSamples | where CookedValue).CookedValue | measure -sum).sum'
        self.gpu_memory_command_process_nonlocal = r'(((Get-Counter "\GPU Process Memory(*)\Non Local Usage").CounterSamples | where CookedValue).CookedValue | measure -sum).sum'
        self.gpu_memory_command_process_committed = r'(((Get-Counter "\GPU Process Memory(*)\Total Committed").CounterSamples | where CookedValue).CookedValue | measure -sum).sum'
        self.gpu_memory_command_engine_utilization_compute = r'(((Get-Counter "\GPU Engine(*engtype_Compute)\Utilization Percentage").CounterSamples | where CookedValue).CookedValue | measure -sum).sum'
        self.gpu_memory_command_engine_utilization_copy = r'(((Get-Counter "\GPU Engine(*engtype_Copy)\Utilization Percentage").CounterSamples | where CookedValue).CookedValue | measure -sum).sum'

        self.peak_sys_memory_committed_bytes = 0.0
        self.peak_sys_memory_commit_limit = 0.0
        self.memory_percentage = 0.0

        self.sys_memory_command_committed_bytes = r'(((Get-Counter "\Memory\Committed Bytes").CounterSamples | where CookedValue).CookedValue | measure -sum).sum'
        self.sys_memory_command_commit_limit = r'(((Get-Counter "\Memory\Commit Limit").CounterSamples | where CookedValue).CookedValue | measure -sum).sum'

        self.peak_memory_lock = threading.Lock()
        self.stop_monitoring = False
        self.IS_GPU_SYSTEM = False
        self.monitor_metrics = monitor
        
        if self.monitor_metrics:
            self._initial_check()   

    
    def __enter__(self):
        if self.monitor_metrics:
            print("Starting memLogger")
            if self.IS_GPU_SYSTEM:
                self.threads_list = []
                self.threads_list.append(threading.Thread(target=self.monitor_sys_memory_command_committed_bytes))
                self.threads_list.append(threading.Thread(target=self.monitor_sys_memory_command_commit_limit))
                self.threads_list.append(threading.Thread(target=self.monitor_gpu_memory_engine_utilization_copy))
                self.threads_list.append(threading.Thread(target=self.monitor_gpu_memory_engine_utilization_compute))
                self.threads_list.append(threading.Thread(target=self.monitor_gpu_memory_command_process_local))
                self.threads_list.append(threading.Thread(target=self.monitor_gpu_memory_command_process_shared))
                self.threads_list.append(threading.Thread(target=self.monitor_gpu_memory_command_process_dedicated))
                self.threads_list.append(threading.Thread(target=self.monitor_gpu_memory_command_process_nonlocal))
                self.threads_list.append(threading.Thread(target=self.monitor_gpu_memory_command_process_committed))
            else:
                raise SystemError

            for th in self.threads_list:
                th.start()
            
        return self     

    
    def __exit__(self, exec_type, exec_val, exec_tb):
        if self.monitor_metrics:
            print("Stopping memLogger")
            time.sleep(5)
            self.stop_monitoring = True
            for th in self.threads_list:
                th.join()

    def _initial_check(self):
        print("Initialization for GPU checker")
        try:
            _ = subprocess.run(['powershell', '-Command', self.gpu_memory_command_process_local], capture_output=True)
            self.IS_GPU_SYSTEM = True
        except Exception:
            self.IS_GPU_SYSTEM = False
        print(f"IS_GPU_SYSTEM = {self.IS_GPU_SYSTEM}")
    
    def monitor_sys_memory_command_committed_bytes(self):
        while not self.stop_monitoring:
            result_committed_bytes = subprocess.run(['powershell', '-Command', self.sys_memory_command_committed_bytes], capture_output=True).stdout.decode("ascii")
            memory_usage_committed_bytes = float(result_committed_bytes.strip().replace(',', '.'))
            sys_memory_committed_bytes = memory_usage_committed_bytes
            current_peak_committed_bytes = round(sys_memory_committed_bytes / 1024**3, 2)

            with self.peak_memory_lock:
                self.peak_sys_memory_committed_bytes = max(current_peak_committed_bytes, self.peak_sys_memory_committed_bytes)
            time.sleep(0.1)

    def monitor_sys_memory_command_commit_limit(self):
        while not self.stop_monitoring:
            result_commit_limit = subprocess.run(['powershell', '-Command', self.sys_memory_command_commit_limit], capture_output=True).stdout.decode("ascii")
            memory_usage_commit_limit = float(result_commit_limit.strip().replace(',', '.'))
            sys_memory_commit_limit = memory_usage_commit_limit
            current_peak_commit_limit = round(sys_memory_commit_limit / 1024**3, 2)

            with self.peak_memory_lock:
                self.peak_sys_memory_commit_limit = max(current_peak_commit_limit, self.peak_sys_memory_commit_limit)
            time.sleep(0.1)

    def monitor_gpu_memory_engine_utilization_copy(self):
        while not self.stop_monitoring:
            
            result_engine_utilization_copy = subprocess.run(['powershell', '-Command', self.gpu_memory_command_engine_utilization_copy], capture_output=True).stdout.decode("ascii")
            memory_usage_engine_utilization_copy = float(result_engine_utilization_copy.strip().replace(',', '.'))
            gpu_memory_engine_utilization_copy = memory_usage_engine_utilization_copy
            current_peak_engine_utilization_copy = round(gpu_memory_engine_utilization_copy, 2)

            self.peak_gpu_memory_engine_utilization_copy = max(current_peak_engine_utilization_copy, self.peak_gpu_memory_engine_utilization_copy)
        time.sleep(0.1)

    def monitor_gpu_memory_engine_utilization_compute(self):
        while not self.stop_monitoring:
            
            result_engine_utilization_compute = subprocess.run(['powershell', '-Command', self.gpu_memory_command_engine_utilization_compute], capture_output=True).stdout.decode("ascii")
            memory_usage_engine_utilization_compute = float(result_engine_utilization_compute.strip().replace(',', '.'))
            gpu_memory_engine_utilization_compute = memory_usage_engine_utilization_compute
            current_peak_engine_utilization_compute = round(gpu_memory_engine_utilization_compute, 2)
            
            self.peak_gpu_memory_engine_utilization_compute = max(current_peak_engine_utilization_compute, self.peak_gpu_memory_engine_utilization_compute)
        time.sleep(0.1)
        
    def monitor_gpu_memory_command_process_local(self):
        while not self.stop_monitoring:
            
            result_process_local = subprocess.run(['powershell', '-Command', self.gpu_memory_command_process_local], capture_output=True).stdout.decode("ascii")
            memory_usage_process_local = float(result_process_local.strip().replace(',', '.'))
            gpu_memory_process_local = memory_usage_process_local
            current_peak_process_local = round(gpu_memory_process_local / 1024**3, 2)
            
            self.peak_gpu_memory_process_local = max(current_peak_process_local, self.peak_gpu_memory_process_local)
        time.sleep(0.1)
        
    def monitor_gpu_memory_command_process_shared(self):
        while not self.stop_monitoring:
            
            result_process_shared = subprocess.run(['powershell', '-Command', self.gpu_memory_command_process_shared], capture_output=True).stdout.decode("ascii")
            memory_usage_process_shared = float(result_process_shared.strip().replace(',', '.'))
            gpu_memory_process_shared = memory_usage_process_shared
            current_peak_process_shared = round(gpu_memory_process_shared / 1024**3, 2)
            
            self.peak_gpu_memory_process_shared = max(current_peak_process_shared, self.peak_gpu_memory_process_shared)
        time.sleep(0.1)
        
    def monitor_gpu_memory_command_process_dedicated(self):
        while not self.stop_monitoring:
            
            result_process_dedicated = subprocess.run(['powershell', '-Command', self.gpu_memory_command_process_dedicated], capture_output=True).stdout.decode("ascii")
            memory_usage_process_dedicated = float(result_process_dedicated.strip().replace(',', '.'))
            gpu_memory_process_dedicated = memory_usage_process_dedicated
            current_peak_process_dedicated = round(gpu_memory_process_dedicated / 1024**3, 2)
            
            self.peak_gpu_memory_process_dedicated = max(current_peak_process_dedicated, self.peak_gpu_memory_process_dedicated)
        time.sleep(0.1)
        
    def monitor_gpu_memory_command_process_nonlocal(self):
        while not self.stop_monitoring:
            
            result_process_nonlocal = subprocess.run(['powershell', '-Command', self.gpu_memory_command_process_nonlocal], capture_output=True).stdout.decode("ascii")
            memory_usage_process_nonlocal = float(result_process_nonlocal.strip().replace(',', '.'))
            gpu_memory_process_nonlocal = memory_usage_process_nonlocal
            current_peak_process_nonlocal = round(gpu_memory_process_nonlocal / 1024**3, 2)
            
            self.peak_gpu_memory_process_nonlocal = max(current_peak_process_nonlocal, self.peak_gpu_memory_process_nonlocal)
        time.sleep(0.1)
        
    def monitor_gpu_memory_command_process_committed(self):
        while not self.stop_monitoring:
            
            result_process_committed = subprocess.run(['powershell', '-Command', self.gpu_memory_command_process_committed], capture_output=True).stdout.decode("ascii")
            memory_usage_process_committed = float(result_process_committed.strip().replace(',', '.'))
            gpu_memory_process_committed = memory_usage_process_committed
            current_peak_process_committed = round(gpu_memory_process_committed / 1024**3, 2)
            
            self.peak_gpu_memory_process_committed = max(current_peak_process_committed, self.peak_gpu_memory_process_committed)
        time.sleep(0.1)
            
    def print_summary(self):
        if self.monitor_metrics:
            if self.IS_GPU_SYSTEM:
                self.memory_percentage = round((self.peak_sys_memory_committed_bytes/self.peak_sys_memory_commit_limit)*100, 2)
                print(" GPU Info ".center(80, '='))
                print(f"Peak GPU Memory Usage (Local): {self.peak_gpu_memory_process_local} GiB")
                print(f"Peak GPU Memory Usage (Shared): {self.peak_gpu_memory_process_shared} GiB")
                print(f"Peak GPU Memory Usage (Dedicated): {self.peak_gpu_memory_process_dedicated} GiB")
                print(f"Peak GPU Memory Usage (NonLocal): {self.peak_gpu_memory_process_nonlocal} GiB")
                print(f"Peak GPU Memory Usage (Committed): {self.peak_gpu_memory_process_committed} GiB")
                print(f"Peak GPU Engine Usage (Compute): {self.peak_gpu_memory_engine_utilization_compute} %")
                print(f"Peak GPU Engine Usage (Copy): {self.peak_gpu_memory_engine_utilization_copy} %")
                print(" Sys Info ".center(80, '='))
                print(f"Peak Sys Memory Usage: {self.peak_sys_memory_committed_bytes} GiB ({self.memory_percentage} %) / {self.peak_sys_memory_commit_limit} GiB")


def workload():
    from pathlib import Path

    import numpy as np
    from optimum.intel.openvino import OVDiffusionPipeline
    from PIL import Image

    model_id = "nmkd/stable-diffusion-1.5-fp16"
    model_dir = Path("sd_v1.5")
    prompt = "sailing ship in storm by Rembrandt"
    num_steps = 20
    gscale = 7.5
    height = 512
    width = 512
    batch_size = 1

    def _generate_prompts(batch_size=1):
        inputs = {
            "prompt": [prompt] * batch_size,
            "num_inference_steps": num_steps,
            "guidance_scale": gscale,
            "output_type": "np",
        }
        return inputs

    def generate_inputs(height=128, width=128, batch_size=1):
        inputs = _generate_prompts(batch_size=batch_size)

        inputs["height"] = height
        inputs["width"] = width

        return inputs

    if not model_dir.exists():
        ov_pipe = OVDiffusionPipeline.from_pretrained(model_id, device="GPU")
        ov_pipe.save_pretrained(model_dir)
    else:
        ov_pipe = OVDiffusionPipeline.from_pretrained(model_dir, device="GPU") 


    #GPU here assumes only 1 GPU exists. If multi-GPU or hybrid, need to specifically target GPU with GPU.0/GPU.1 etc

    inputs = generate_inputs(height, width, batch_size)
    print("Start workload measure")
    start = time.time()
    image = ov_pipe(**inputs).images
    stop = time.time()
    print("Stop workload measure")
    print(" Workload Info ".center(80, '='))
    print(f"Inference time: {round(stop-start,2)} s")
    for img_idx in range(image.shape[0]):
        img = Image.fromarray(np.uint8(image[img_idx,:,:,:] * 255)).convert("RGB")
        img.save(f"test{img_idx}.png")      

        
def main():
    with MemLogger() as mlg:
        workload()
    mlg.print_summary()

    
if __name__ == "__main__":
    main()

