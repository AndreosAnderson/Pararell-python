import numpy as np
import pytest
import threading
import time
from threading import Thread
import psutil
from memory_profiler import memory_usage

def track_memory(func, *args):
    mem_usage = memory_usage((func, args))
    print(f"Memory usage: {max(mem_usage)} MB")
    return mem_usage


def track_cpu(func, *args):
    cpu_percent_before = psutil.cpu_percent(interval=1)
    start_time = time.time()

    func(*args)

    end_time = time.time()
    cpu_percent_after = psutil.cpu_percent(interval=1)

    print(f"Initial CPU usage: {cpu_percent_before}%")
    print(f"Final CPU usage: {cpu_percent_after}%")
    print(f"Execution time: {end_time - start_time} seconds")

def process(A,B,C,N,startRow,endRow):
    for i in range(startRow,endRow,1):
        for j in range(0,N,1):
            for k in range(0,N,1):
                C[i,j]+=A[i,j]*B[k,j]

def matrix_multiply(A,B,C,N,numThreads):
    tasks=[]
    rowsPerThread = int(N/numThreads)
    remainingRows = int(N%numThreads)

    startRow=0
    for i in range(0,numThreads,1):
        endRow = startRow+rowsPerThread
        if i<remainingRows:
            endRow+=1
        tasks.append(Thread(target=process, args=(A,B,C,N,startRow,endRow)))
        tasks[i].start()
        startRow = endRow
    for i in range(0,numThreads,1):
        tasks[i].join()
    
@pytest.fixture
def setup_matrices():
    size = 1024
    numThreads=16
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    C = np.zeros((size,size),dtype = float)
    return A,B,C,size,numThreads

@pytest.mark.benchmark(min_rounds=5)
def test_matrix_multiply(benchmark, setup_matrices):
    A, B, C, N, numThreads = setup_matrices

    result = benchmark(matrix_multiply, A, B, C, N, numThreads)
    track_memory(matrix_multiply, A, B, C, N, numThreads)
    track_cpu(matrix_multiply, A, B, C, N, numThreads)

    assert np.any(C != 0), "Matrix C has not been updated."