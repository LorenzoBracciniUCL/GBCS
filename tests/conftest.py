import os
import sys

# Before numpy loads.  The solvers work on very small matrices, where BLAS threads only add contention.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(variable, '1')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
