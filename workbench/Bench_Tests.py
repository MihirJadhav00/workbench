import csv
import gc
import os
import time
import warnings
from datetime import datetime

import numpy as np
import plotly.graph_objects as px
from plotly.subplots import make_subplots
from memory_profiler import memory_usage

from stingray import (AveragedCrossspectrum, AveragedPowerspectrum,
                      Crossspectrum, Lightcurve, Powerspectrum)

warnings.filterwarnings("ignore")


def benchCode(benchFunc, *args):
    """
    Generalized code for benchmarking time and memory.

    Parameters
    ----------
    benchFunc : function
        Function to be benchmarked.

    Returns
    -------
    time1 : float
        Time benchmark of the function.

    mem1 : float
        Memory footprint of the function.
    """
    gc.disable()
    start = time.perf_counter()
    temp = [benchFunc(*args) for i in range(3)]
    time1 = (time.perf_counter() - start) / 5
    gc.enable()
    del temp

    mem1 = memory_usage((benchFunc, args))

    return time1, sum(mem1) / len(mem1)


def CSVWriter(func_dict, wall_time, mem_use):
    """
    Write the benchmark results to a CSV file.

    Parameters
    ----------
    func_dict : dict
        Dictionary of all the functions to be saved.

    wall_time : list
        List of wall times for all functions.

    mem_use : list
        List of memory use for all functions.
    """
    for class_name, funcs in func_dict.items():
        for i, func_name in enumerate(funcs):
            if not os.path.isfile(f'workbench/workbench/data/{class_name}/{func_name}.csv'):
                os.makedirs(f'workbench/workbench/data/{class_name}', exist_ok=True)
                with open(f'workbench/workbench/data/{class_name}/{func_name}.csv', 'w+') as fptr:
                    writer = csv.writer(fptr)
                    writer.writerow([
                        'Commit_Tstamp', 'Commit_Msg', '100K', '1M', '10M',
                        '100M', '1B'
                    ])

            with open(f'workbench/workbench/data{class_name}/{func_name}.csv', 'a+') as fptr:
                writer = csv.writer(fptr)

                if func_name[0] == 'T':
                    writer.writerow(wall_time[int(i / 2)])

                elif func_name[0] == 'M':
                    writer.writerow(mem_use[int(i / 2)])


def CSVPlotter(path, class_name, func_name):
    """
    Plots results of benchmarks of a given function from a CSV file.

    Parameters
    ----------
    path: string
        Path to find files.

    class_name : string
        Name of the folder/class whose function is to be plotted.

    func_name : string
        Name of the file/function to be plotted.
    """
    graph_val = ['100K', '1M', '10M', '100M', '1B']
    for root, dirs, files in os.walk(path):
        if root[root.rfind('/', 0, len(root)) + 1:len(root)] == class_name and f'Time_{func_name}.csv' in files:
            T_file = os.path.join(root, f'Time_{func_name}.csv')
            M_file = os.path.join(root, f'Mem_{func_name}.csv')
    fig = make_subplots(rows=1,
                        cols=2,
                        subplot_titles=("Execution Time(in s)(log)",
                                        "Memory Use(in MB)(log)"))

    with open(T_file, 'r+') as T_ptr:
        reader = csv.reader(T_ptr)
    
        for count, row in enumerate(reader):
            if count != 0 and row:
                fig.add_trace(px.Scatter(x=graph_val, y=row[2:], name=f'{row[0]}-{row[1]}'), row=1, col=1)
                fig.update_yaxes(type="log", row=1, col=1)

    with open(M_file, 'r+') as M_ptr:
        reader = csv.reader(M_ptr)

        for count, row in enumerate(reader):
            if count != 0 and row:
                fig.add_trace(px.Scatter(x=graph_val, y=row[2:], name=f'{row[0]}-{row[1]}'), row=1, col=2)
                fig.update_yaxes(type="log", row=1, col=2)

    fig.update_layout(title_text=f"Benchmark for {func_name}")
    fig.show()


def makeLCFunc(times):
    Lightcurve.make_lightcurve(times, dt=1.0)


def createLc(times, counts):
    Lightcurve(times, counts)


def createLcP(times, counts):
    Lightcurve(times, counts, dt=1.0, skip_checks=True)


def lcMJD(lc_obj):
    lc_obj.change_mjdref(-2379826)


def rebinSum(lc_obj, dt_new):
    lc_obj.rebin(dt_new)


def rebinMean(lc_obj, dt_new):
    lc_obj.rebin(dt_new, method='mean')


def addLC(lc1, lc2):
    lc1.__add__(lc2)


def subLC(lc1, lc2):
    lc1.__sub__(lc2)


def eqLC(lc1, lc2):
    lc1.__eq__(lc2)


def negLC(lc1):
    lc1.__neg__()


def indexTrunc(lc):
    lc.truncate(0, lc.__len__() // 2)


def tTrunc(lc):
    lc.truncate(0, lc.__len__() // 2, method='time')


def splitLc(lc, min_gap):
    lc.split(min_gap)


def sortLcTime(lc):
    lc.sort()


def sortLcCount(lc):
    lc.sort_counts()


def chunkAnlyze(lc, chunk_len, target_func):
    lc.analyze_lc_chunks(chunk_len, target_func)


def chunkLen(lc, min_count, min_t_bins):
    lc.estimate_chunk_length(min_count, min_t_bins)


def joinLc(lc1, lc2):
    lc1.join(lc2)


def createCspec(lc1, lc2):
    Crossspectrum(lc1, lc2, dt=1.0)


def rebinCspec(cspec):
    cspec.rebin(df=2.0)


def coherCspec(cspec):
    cspec.coherence()


def TlagCspec(cspec):
    cspec.time_lag()


def createAvgCspec(lc1, lc2, seg):
    AveragedCrossspectrum(lc1, lc2, seg, silent=True)


def coherAvgCspec(avg_Cspec):
    avg_Cspec.coherence()


def TlagAvgCspec(avg_Cspec):
    avg_Cspec.time_lag()


def createPspec(lc):
    Powerspectrum(lc)


def rebinPspec(pspec):
    pspec.rebin(df=0.01)


def classSign(pspec):
    pspec.classical_significances()


def pspecRMS(pspec):
    pspec.compute_rms(min_freq=min(pspec.freq) * 10,
                      max_freq=max(pspec.freq) / 1.5)


def createAvgPspec(lc, seg):
    AveragedPowerspectrum(lc, seg)


def callerFunction(bench_msg):
    # bench_msg = input("Enter the changes made, if none put a '-': ")
    func_dict = {
        'Lightcurve': [
            'Time_MakeLightcurve', 'Mem_MakeLightcurve', 'Time_InitNoParam',
            'Mem_InitNoParam', 'Time_InitParam', 'Mem_InitParam',
            'Time_ChangeMJDREF', 'Mem_ChangeMJDREF', 'Time_Rebin_Sum',
            'Mem_Rebin_Sum', 'Time_Rebin_Mean_Avg', 'Mem_Rebin_Mean_Avg',
            'Time_AddLC', 'Mem_AddLC', 'Time_SubLC', 'MemSubLC', 'Time_EqLC',
            'Mem_EqLC', 'Time_NegLC', 'Mem_NegLC', 'Time_Trunc_Index',
            'Mem_Trunc_Index', 'Time_Trunc_Time', 'Mem_Trunc_Time',
            'Time_SplitLC', 'Mem_SplitLC', 'Time_Sort_Time', 'Mem_Sort_Time',
            'Time_Sort_Counts', 'Mem_Sort_Counts', 'Time_Analyze_Chunks',
            'Mem_Analyze_Chunks', 'Time_Est_Chunk_Len', 'Mem_Est_Chunk_Len',
            'Time_JoinLC', 'Mem_JoinLC'
        ],
        'Crossspectrum': [
            'Time_Init', 'Mem_Init', 'Time_Rebin_Linear', 'Mem_Rebin_Linear',
            'Time_Coherence', 'Mem_Coherence', 'Time_Tlag', 'Mem_Tlag'
        ],
        'AveragedCrossspectrum': [
            'Time_Init', 'Mem_Init', 'Time_Coher', 'Mem_Coher', 'Time_Tlag',
            'Mem_Tlag'
        ],
        'Powerspectrum': [
            'Time_Init', 'Mem_Init', 'Time_Rebin', 'Mem_Rebin', 'Time_RMS', 'Mem_RMS', 'Time_Class_Sign', 'Mem_Class_Sign'
        ],
        'AveragedPowerspectrum': ['Time_Init', 'Mem_Init']
    }

    wall_time = [[
        f'{datetime.utcfromtimestamp(int(time.time())).strftime("%Y-%m-%d %H:%M:%S")}',
        f'{bench_msg}',
    ] for i in range(int(sum([len(x) for x in func_dict.values()]) / 2))]
    mem_use = [[
        f'{datetime.utcfromtimestamp(int(time.time())).strftime("%Y-%m-%d %H:%M:%S")}',
        f'{bench_msg}',
    ] for i in range(int(sum([len(x) for x in func_dict.values()]) / 2))]

    for size in [10**i for i in range(5, 7)]:
        num_func = 0
        times = np.arange(size)
        counts = np.random.rand(size) * 100

        time1, mem1 = benchCode(makeLCFunc, times)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        time1, mem1 = benchCode(createLc, times, counts)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        time1, mem1 = benchCode(createLcP, times, counts)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        lc = Lightcurve(times, counts, dt=1.0, skip_checks=True)

        time1, mem1 = benchCode(lcMJD, lc)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        time1, mem1 = benchCode(rebinSum, lc, 2.0)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        time1, mem1 = benchCode(rebinMean, lc, 2.0)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        lc_other = Lightcurve(times,
                              counts * np.random.rand(size),
                              dt=1.0,
                              skip_checks=True)

        time1, mem1 = benchCode(addLC, lc, lc_other)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        time1, mem1 = benchCode(subLC, lc, lc_other)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        time1, mem1 = benchCode(eqLC, lc, lc_other)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1
        del lc_other

        time1, mem1 = benchCode(negLC, lc)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        time1, mem1 = benchCode(indexTrunc, lc)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        time1, mem1 = benchCode(tTrunc, lc)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        times2 = np.arange(0, size, np.random.randint(4, 9))
        counts2 = np.random.rand(len(times)) * 100
        lc_temp = Lightcurve(times, counts, dt=1.0, skip_checks=True)
        time1, mem1 = benchCode(splitLc, lc_temp, 4)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1
        del times2, counts2, lc_temp

        time1, mem1 = benchCode(sortLcTime, lc)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        time1, mem1 = benchCode(sortLcCount, lc)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        time1, mem1 = benchCode(chunkAnlyze, lc, 100000, lambda x: np.mean(x))
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        time1, mem1 = benchCode(chunkLen, lc, 10000, 10000)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        lc_other = Lightcurve(times,
                              counts * np.random.rand(size),
                              dt=1.0,
                              skip_checks=True)

        time1, mem1 = benchCode(joinLc, lc, lc_other)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        time1, mem1 = benchCode(createCspec, lc, lc_other)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        cspec = Crossspectrum(lc, lc_other, dt=1.0)

        time1, mem1 = benchCode(rebinCspec, cspec)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        time1, mem1 = benchCode(coherCspec, cspec)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        time1, mem1 = benchCode(TlagCspec, cspec)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        del cspec

        time1, mem1 = benchCode(createAvgCspec, lc, lc_other, 10000)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        avg_cspec = AveragedCrossspectrum(lc, lc_other, 10000, silent=True)

        time1, mem1 = benchCode(coherAvgCspec, avg_cspec)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        time1, mem1 = benchCode(TlagAvgCspec, avg_cspec)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        del avg_cspec, lc_other

        time1, mem1 = benchCode(createPspec, lc)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        pspec = Powerspectrum(lc)

        time1, mem1 = benchCode(rebinPspec, pspec)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        time1, mem1 = benchCode(pspecRMS, pspec)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        temp_pspec = Powerspectrum(lc, norm='leahy')
        time1, mem1 = benchCode(classSign, temp_pspec)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        del pspec, temp_pspec

        time1, mem1 = benchCode(createAvgPspec, lc, 10000)
        wall_time[num_func].append(time1)
        mem_use[num_func].append(mem1)
        num_func += 1

        del lc, time1, mem1
    CSVWriter(func_dict, wall_time, mem_use)
    del func_dict, wall_time, mem_use

   
