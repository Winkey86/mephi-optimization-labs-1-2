"""Variant 17: bakery LP, shadow prices, dual, elastic filter and LP benchmark."""
import json
import platform
from common import ROOT, SEED, TOL, np, plt, solve, meta, save_json, csv_write, finish_plot


def load_data():
    return json.loads((ROOT / 'lab1/data/variant.json').read_text())


def verify(d, x, b, objective):
    # Recompute from domain input, without using a solver model or its residuals.
    x, b = np.asarray(x, float), np.asarray(b, float)
    if x.shape != (8,) or b.shape != (3,):
        raise ValueError('Expected eight product quantities and three oven capacities')
    if not (np.isfinite(x).all() and np.isfinite(b).all() and np.isfinite(objective)):
        raise ValueError('Solution, capacities and objective must be finite')
    A = np.array(d['oven_hours_per_batch'])
    low, high = np.array(d['min_batches']), np.array(d['max_batches'])
    used = np.sum(A * x[None, :], axis=1)
    obj = float(np.sum(np.array(d['margin_rub_per_batch']) * x))
    violation = float(max(0, np.max(used - b), np.max(low - x), np.max(x - high)))
    out = {'max_violation': violation, 'objective_recomputed': obj,
           'objective_error': abs(obj - objective), 'oven_usage_h': used,
           'slack_h': b - used, 'tolerance': TOL}
    if violation > TOL or abs(obj - objective) > TOL:
        raise ValueError(f'Independent verification failed: {out}')
    return out


def run():
    d = load_data()
    out = ROOT / 'lab1/results'; out.mkdir(parents=True, exist_ok=True)
    A, b, c, low, high = map(np.array, [d['oven_hours_per_batch'], d['oven_hours'],
               d['margin_rub_per_batch'], d['min_batches'], d['max_batches']])
    scenarios = {'base': b.copy(), 'deficit': b.copy(), 'stress': b.copy()}
    base = solve(c, A, [-np.inf]*3, b, low, high, maximize=True, log=out/'base.log')
    assert base['optimal']
    k = int(np.argmax(base['row_dual']))
    scenarios['deficit'][k] *= .8
    scenarios['stress'][k] = .8 * (A @ low)[k]
    scenarios['stress'][(k+1)%3] *= .5
    results = {}; records = []; checks = {}
    for name, rhs in scenarios.items():
        r = base if name == 'base' else solve(c, A, [-np.inf]*3, rhs, low, high,
                                  maximize=True, log=out/f'{name}.log')
        results[name] = {**meta(r), 'x': r['x'], 'rhs': rhs}
        records.append({'scenario': name, **meta(r)})
        if r['optimal']:
            checks[name] = verify(d, r['x'], rhs, r['objective'])
            csv_write(out/f'{name}_solution.csv', ({'product': p, 'batches_10_units': float(x),
                         'units': float(10*x), 'margin_rub': float(ci*x)}
                         for p,x,ci in zip(d['products'],r['x'],c)))
            csv_write(out/f'{name}_constraints.csv', ({'oven': i+1, 'usage_h': float(v),
                       'capacity_h': float(rhs[i]), 'slack_h': float(rhs[i]-v),
                       'shadow_rub_per_h': float(r['row_dual'][i])}
                       for i,v in enumerate(A @ r['x'])))
    assert results['stress']['status'] == 'Infeasible'
    required = float((A @ low)[k]); available = float(scenarios['stress'][k])
    checks['stress_certificate'] = {'oven': k+1, 'contract_requires_h': required,
             'capacity_h': available, 'unavoidable_shortfall_h': required-available}
    # Explicit dual of shifted primal z=x-low, z<=high-low.
    G = np.vstack([A,np.eye(8)])
    rhs = np.r_[b-A@low, high-low]
    dual = solve(rhs, G.T, c, [np.inf]*8, log=out/'dual.log')
    assert dual['optimal']
    y = dual['x']; z = base['x']-low
    dual_obj = float(c@low + rhs@y)
    dualcheck = {'objective_with_constant': dual_obj,
        'strong_duality_error': abs(dual_obj-base['objective']),
        'dual_violation': float(max(0,np.max(c-G.T@y),-np.min(y))),
        'complementarity_max': float(max(np.max(abs(y*(rhs-G@z))),
                                         np.max(abs(z*(G.T@y-c)))))}
    assert max(dualcheck['strong_duality_error'],dualcheck['dual_violation'],
               dualcheck['complementarity_max']) <= TOL
    checks['dual'] = dualcheck
    csv_write(out/'dual_solution.csv', ({'constraint': f'oven_{i+1}' if i<3 else f'demand_{i-2}',
                   'dual_rub_per_unit': float(v)} for i,v in enumerate(y)))
    shadow = []
    for factor in [.001, .01, .05, -.2]:
        newb = b.copy(); delta = factor*b[k]; newb[k] += delta
        r = solve(c,A,[-np.inf]*3,newb,low,high,maximize=True,log=out/f'shadow_{factor}.log')
        actual = r['objective']-base['objective'] if r['optimal'] else None
        shadow.append({'fraction':factor,'delta_h':delta,
               'forecast_delta_rub':base['row_dual'][k]*delta,
               'actual_delta_rub':actual,
               'prediction_error_rub':None if actual is None else actual-base['row_dual'][k]*delta})
    csv_write(out/'shadow_experiments.csv', shadow)
    # Minimal normalized capacity additions, retaining all contractual minima.
    elastic_A = np.hstack([A,-np.eye(3)])
    elastic = solve(np.r_[np.zeros(8),1/b], elastic_A, [-np.inf]*3,
             scenarios['stress'], np.r_[low,np.zeros(3)], np.r_[high,[np.inf]*3],
             log=out/'elastic.log')
    assert elastic['optimal']
    additions = elastic['x'][8:]
    checks['elastic'] = verify(d,elastic['x'][:8],scenarios['stress']+additions,
                           float(c@elastic['x'][:8]))
    checks['elastic']['capacity_additions_h'] = additions
    checks['elastic']['normalized_objective'] = elastic['objective']
    assert np.max(abs(additions-np.maximum(A@low-scenarios['stress'],0))) <= TOL
    csv_write(out/'elastic.csv',({'oven':i+1,'added_h':v,'normalized_added':v/b[i]}
                               for i,v in enumerate(additions)))
    # Ten increasing LP sizes. Identical generated inputs, 3 repeats after warmup.
    rng = np.random.default_rng(SEED)
    bench = []; timings = []
    solve(c,A,[-np.inf]*3,b,low,high,maximize=True)
    for size in range(1,11):
        n,m = 40*size,15*size
        B = rng.uniform(.1,1,(m,n)); B[rng.random((m,n))<.65]=0
        u=rng.uniform(1,8,n); cost=rng.uniform(1,30,n); cap=B@(u*.4)
        save_json(out/f'benchmark_{n}_input.json',{'seed':SEED,'n':n,'m':m,
                  'A':B,'c':cost,'upper':u,'rhs':cap})
        objectives=[]
        for alg in ['simplex','ipm']:
            trials=[]; iterations=[]
            for rep in range(3):
                r=solve(cost,B,[-np.inf]*m,cap,upper=u,maximize=True,algorithm=alg,
                        log=out/f'benchmark_{n}_{alg}_{rep}.log')
                assert r['optimal']
                xx=r['x']; assert np.max(B@xx-cap)<=TOL and np.min(xx)>=-TOL and np.max(xx-u)<=TOL
                assert abs(cost@xx-r['objective']) <= TOL
                trials.append(r['seconds'])
                iterations.append(r['simplex_iterations'] if alg=='simplex' else r['ipm_iterations'])
                timings.append({'n':n,'m':m,'algorithm':alg,'repeat':rep+1,**meta(r)})
            objectives.append(r['objective'])
            bench.append({'n':n,'m':m,'algorithm':alg,'median_seconds':float(np.median(trials)),
                          'median_iterations':float(np.median(iterations)), 'objective':r['objective']})
        assert abs(objectives[0]-objectives[1]) <= TOL*max(1,abs(objectives[0]))
    csv_write(out/'benchmark.csv',bench); csv_write(out/'benchmark_trials.csv',timings)
    fig,axes=plt.subplots(1,2,figsize=(9,3.4))
    for alg in ['simplex','ipm']:
        q=[r for r in bench if r['algorithm']==alg]
        axes[0].plot([r['n'] for r in q],[r['median_seconds']*1000 for r in q],'-o',label=alg)
        axes[1].plot([r['n'] for r in q],[r['median_iterations'] for r in q],'-o',label=alg)
    for ax in axes: ax.set_xlabel('Число переменных'); ax.grid(alpha=.2); ax.legend()
    axes[0].set_ylabel('Медиана времени, мс'); axes[1].set_ylabel('Итерации своего алгоритма')
    finish_plot(out/'benchmark.png')
    fig,ax=plt.subplots(figsize=(8,3.5)); width=.36; inds=np.arange(8)
    for off,name,label in [(-width/2,'base','Базовый'),(width/2,'deficit','Дефицитный')]:
        ax.bar(inds+off,results[name]['x'],width,label=label)
    ax.set_xticks(inds,d['products'],rotation=20,ha='right'); ax.set_ylabel('Партии по 10 шт.'); ax.legend()
    finish_plot(out/'plan.png')
    summary={'scenarios':results,'constrained_oven':k+1,'shadow_prices':base['row_dual'],
             'shadow_experiments':shadow,'dual':dualcheck,'benchmark':bench,
             'host':platform.platform()}
    save_json(out/'summary.json',summary); save_json(out/'verification.json',checks)
    csv_write(out/'scenarios.csv',records)
    return summary


if __name__ == '__main__':
    print(run()['scenarios'])
