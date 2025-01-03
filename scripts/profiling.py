#!/usr/bin/env python3

import enum
import pandas as pd
import psutil
import time
import threading


class HWDataKey(enum.Enum):
    TIMESTAMP = 0
    MEM_USAGE_PERCENT = 1
    MEM_USAGE_SIZE = 2
    CPU_USAGE_PERCENT = 3

class HWResourceTracker(threading.Thread):
    def __init__(self, process = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.process = process
        self.running = True
        self.data = []
        self.period_time = 0.1  # unit: seconds
        self.lock = threading.Lock()

    def set_property(self, props):
        self.period_time = props.get('period_time', 0.1)

    def run(self):
        while self.running:
            time.sleep(self.period_time)
            timestamp_ms = round(time.time()*1000)

            if (self.process != None):
                if self.process.is_running():
                    self._append_data(timestamp_ms,
                                 self.process.memory_percent(),
                                 self.process.memory_info().rss,
                                 self.process.cpu_percent())
            else:
                memory_usage_dict = dict(psutil.virtual_memory()._asdict())
                self._append_data(timestamp_ms,
                             memory_usage_dict['percent'],
                             memory_usage_dict['used'],
                             psutil.cpu_percent())

    def _append_data(self, *argv):
        self.lock.acquire()
        self.data.append([ value for value in argv ])
        self.lock.release()

    def _get_usage_info(self):
        self.lock.acquire()
        df = pd.DataFrame(self.data, columns=[ key.name for key in HWDataKey ])
        df_max = df.max()
        df_min = df.min()
        df_mean = df.mean()
        self.lock.release()

        return df_mean[HWDataKey.CPU_USAGE_PERCENT.name], \
               df_max[HWDataKey.MEM_USAGE_SIZE.name] - df_min[HWDataKey.MEM_USAGE_SIZE.name], \
               df_max[HWDataKey.MEM_USAGE_PERCENT.name] - df_min[HWDataKey.MEM_USAGE_PERCENT.name]

    def get_data_min_max_last(self, key):
        self.lock.acquire()
        df = pd.DataFrame(self.data, columns=[ dataKey.name for dataKey in HWDataKey ])
        df_max = df.max()
        df_min = df.min()
        df_last = df.iloc[-1]
        self.lock.release()

        return df_min[key.name], df_max[key.name], df_last[key.name]

    def save_graph(self, key, save_path):
        self.lock.acquire()
        df = pd.DataFrame(self.data, columns=[ key.name for key in HWDataKey ])
        df.plot(x=HWDataKey.TIMESTAMP.name, y=key.name, kind='line')
        plt.savefig(save_path)
        self.lock.release()

    def stop(self):
        if self.is_alive():
            self.running = False
            self.join()
            self.running = True

        return self._get_usage_info()
