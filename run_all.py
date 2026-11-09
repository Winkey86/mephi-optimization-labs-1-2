"""Run labs 1 and 2; --generate regenerates the same variant inputs."""
import argparse
import platform
from common import ROOT,save_json,solve,np
import highspy
import numpy
import matplotlib
import lab1,lab2
from generate_data import generate


def self_check():
    # Distinguish solver statuses and make sure the independent checker rejects a bad plan.
    assert solve([1],[[1],[1]],[1,-np.inf],[np.inf,0])['status']=='Infeasible'
    assert solve([-1],[[0]],[-np.inf],[0],presolve=False)['status']=='Unbounded'
    d=lab1.load_data()
    try:
        lab1.verify(d,np.full(8,1e6),np.array(d['oven_hours']),0)
    except ValueError:
        pass
    else:
        raise AssertionError('Independent checker accepted an invalid solution')
    tariff=np.full((4,5,2),1000.0);cap=np.zeros((4,5,2))
    tariff[:2,:2,0]=[[1,2],[2,100]];cap[:2,:2,0]=1
    example={'supply_t':[1,1,0,0],'cost_rub_per_t':tariff}
    flow,unmet,_=lab2.greedy(example,[1,1,0,0,0],cap)
    assert np.max(unmet)==0 and np.sum(tariff*flow)==101
    assert tariff[0,1,0]+tariff[1,0,0]==4


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true')
    args=parser.parse_args()
    if args.generate:generate()
    self_check()
    for label,lab in [('ЛР1',lab1),('ЛР2',lab2)]:
        lab.run();print(f'{label}: расчёт и независимая проверка завершены',flush=True)
    save_json(ROOT/'environment.json',{'python':platform.python_version(),
          'highspy':highspy.Highs().version(),'numpy':numpy.__version__,
          'matplotlib':matplotlib.__version__,'system':platform.platform(),
          'seed':10119,'threads':1,'tolerance':1e-7})
