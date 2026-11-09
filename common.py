"""HiGHS interface and file output. Domain checks live in each lab, not here."""
import csv
import json
import os
import time
from pathlib import Path

import highspy as hs
import numpy as np

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent / '.mplcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 10119
TOL = 1e-7
ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 200})


def save_json(path, data):
    def convert(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    default=convert, allow_nan=False), encoding='utf-8')


def csv_write(path, rows):
    rows = list(rows)
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def solve(c, A, lo, hi, lower=None, upper=None, integer=(), log=None,
          maximize=False, algorithm='simplex', presolve=True):
    c, A = np.asarray(c, float), np.asarray(A, float)
    n = len(c)
    if A.ndim != 2 or A.shape[1] != n:
        raise ValueError('Constraint matrix and objective dimensions do not match')
    if not (np.isfinite(A).all() and np.isfinite(c).all()):
        raise ValueError('Model coefficients must be finite')
    h = hs.Highs()
    options = {'log_to_console': False, 'threads': 1, 'random_seed': SEED,
               'presolve': 'on' if presolve else 'off', 'solver': algorithm,
               'primal_feasibility_tolerance': 1e-8,
               'dual_feasibility_tolerance': 1e-8,
               'mip_feasibility_tolerance': 1e-8, 'mip_rel_gap': 0.0,
               'time_limit': 120.0}
    if log:
        log = Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.unlink(missing_ok=True)
        options['log_file'] = str(log)
    else:
        options['output_flag'] = False
    for key, val in options.items():
        if h.setOptionValue(key, val) != hs.HighsStatus.kOk:
            raise RuntimeError(f'HiGHS rejected option {key}')
    lp = hs.HighsLp()
    lp.num_col_, lp.num_row_ = n, len(A)
    lp.col_cost_ = c
    lp.col_lower_ = np.zeros(n) if lower is None else np.asarray(lower, float)
    lp.col_upper_ = np.full(n, hs.kHighsInf) if upper is None else np.asarray(upper, float)
    lp.row_lower_, lp.row_upper_ = np.asarray(lo, float), np.asarray(hi, float)
    lp.sense_ = hs.ObjSense.kMaximize if maximize else hs.ObjSense.kMinimize
    lp.a_matrix_.format_ = hs.MatrixFormat.kRowwise
    rows, cols = np.nonzero(A)
    lp.a_matrix_.start_ = np.r_[0, np.cumsum(np.count_nonzero(A, axis=1))].astype(np.int32)
    lp.a_matrix_.index_ = cols.astype(np.int32)
    lp.a_matrix_.value_ = A[rows, cols]
    if h.passModel(lp) == hs.HighsStatus.kError:
        raise RuntimeError('HiGHS rejected the model')
    if len(integer):
        inds = np.asarray(integer, np.int32)
        status = h.changeColsIntegrality(len(inds), inds,
                   np.array([hs.HighsVarType.kInteger] * len(inds)))
        if status != hs.HighsStatus.kOk:
            raise RuntimeError('HiGHS rejected integer variable types')
        if h.setOptionValue('solver', 'choose') != hs.HighsStatus.kOk:
            raise RuntimeError('HiGHS rejected the MILP solver option')
    start = time.perf_counter()
    if h.run() == hs.HighsStatus.kError:
        raise RuntimeError('HiGHS failed while solving the model')
    elapsed = time.perf_counter() - start
    info, sol = h.getInfo(), h.getSolution()
    optimal = h.getModelStatus() == hs.HighsModelStatus.kOptimal
    result = {'status': h.modelStatusToString(h.getModelStatus()), 'optimal': optimal,
              'seconds': elapsed, 'simplex_iterations': info.simplex_iteration_count,
              'ipm_iterations': info.ipm_iteration_count,
              'nodes': info.mip_node_count if len(integer) else 0,
              'mip_gap': info.mip_gap if len(integer) and optimal else None,
              'objective': info.objective_function_value if optimal else None,
              'x': np.array(sol.col_value) if optimal else None,
              'row_dual': np.array(sol.row_dual) if optimal and not len(integer) else None,
              'col_dual': np.array(sol.col_dual) if optimal and not len(integer) else None}
    if log:
        if h.setOptionValue('log_file', '') != hs.HighsStatus.kOk:
            raise RuntimeError('HiGHS failed to close the solver log')
        if h.writeModel(str(log.with_suffix('.lp'))) == hs.HighsStatus.kError:
            raise RuntimeError('HiGHS failed to export the model')
    return result


def meta(r):
    return {k: v for k, v in r.items() if k not in ('x', 'row_dual', 'col_dual')}


def finish_plot(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
