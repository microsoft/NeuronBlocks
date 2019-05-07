# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import multiprocessing
from multiprocessing import cpu_count
import math

class ProcessorsScheduler(object):
    process_num = cpu_count()

    def __init__(self, cpu_num_workers=None):
        if cpu_num_workers != None and cpu_num_workers > 0:
            self.process_num = cpu_num_workers

    def run_data_parallel(self, func, func_args):
        data, rest_args =  func_args[0], func_args[1:]
        res = []
        # logging.info("multiprocess enabled, process num: %d" % (self.process_num))
        process_p = multiprocessing.Pool(self.process_num)
        data_length = len(data)
        size = math.ceil(data_length/ self.process_num)
        
        for i in range(self.process_num):
            start = size * i
            end = (i + 1) * size if (i + 1) * size < data_length else data_length
            args = (data[start:end], ) + rest_args
            res.append((i, process_p.apply_async(func, args=args)))
        process_p.close()
        process_p.join()
        res = sorted(res, key=lambda x:x[0])
        return res
