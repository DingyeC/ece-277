import sys
sys.path.append("../build/src/pybind11_cuda_madd/release")
sys.path.append("../build/src/pybind11_cuda_mmul/release")
sys.path.append("../build/src/pybind11_cuda_mmac/release")

import time
import numpy as np

import cu_matrix_madd
import cu_matrix_mmul
import cu_matrix_mmac

M = 32*32
N = 64
k = 64
KIJ = 9

print('nij =',M)
print('input channel =',N)
print('output channel =',k)
print('kij =',KIJ)

x=np.random.randint(0,10,(M,k))
y = []
for i in range(KIJ):
    y.append(np.random.randint(0,10,(k,N)))

z=np.zeros((M,N))

cpu_starttime = time.perf_counter()
for i in range(M):
    for j in range(N):
        tmp=0
        for p in range(k):
            for kij in range(KIJ):
                tmp+=x[i][p]*y[kij][p][j]
        z[i][j]=tmp
cpu_endtime = time.perf_counter()

test = cu_matrix_mmul.mmul(x, y[0])

gpu_z=np.zeros((M,N))
gpu_starttime = time.perf_counter()
for kij in range(KIJ):
    tmp = cu_matrix_mmul.mmul(x, y[kij])
    gpu_z = cu_matrix_madd.madd(gpu_z, tmp)
gpu_endtime = time.perf_counter()

gpu_zm=np.zeros((M,N))
gpu_starttime_m = time.perf_counter()
for kij in range(KIJ):
    gpu_zm = cu_matrix_mmac.mmac(x, y[kij], gpu_zm)
gpu_endtime_m = time.perf_counter()

if (z==gpu_z).all() and (z==gpu_zm).all():
    print('Pass')
else:
    print('Fail')

print('Python Time:',(cpu_endtime-cpu_starttime)*1000,'ms')
print('GPU Time:',(gpu_endtime-gpu_starttime)*1000,'ms')
print('GPU Time(MAC):',(gpu_endtime_m-gpu_starttime_m)*1000,'ms')