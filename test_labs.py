"""Regression checks for solver setup, independent validation and saved variant results."""
import csv
import json
import subprocess
import sys
import unittest

import numpy as np

import lab1
import lab2
from common import ROOT, solve


class SolverTests(unittest.TestCase):
    def test_lp_and_integer_solution(self):
        lp=solve([1],[[2]],[-np.inf],[3],maximize=True)
        integer=solve([1],[[2]],[-np.inf],[3],maximize=True,integer=[0])
        self.assertTrue(lp['optimal'] and integer['optimal'])
        self.assertAlmostEqual(lp['objective'],1.5)
        self.assertAlmostEqual(integer['objective'],1)

    def test_statuses(self):
        self.assertEqual(solve([1],[[1],[1]],[1,-np.inf],[np.inf,0])['status'],'Infeasible')
        self.assertEqual(solve([-1],[[0]],[-np.inf],[0],presolve=False)['status'],'Unbounded')

    def test_nonfinite_model_is_rejected(self):
        with self.assertRaises(ValueError):
            solve([np.nan],[[1]],[-np.inf],[1])

    def test_solver_setup_survives_python_optimization(self):
        program="""
from common import solve
r=solve([1],[[2]],[-float('inf')],[3],maximize=True,integer=[0])
if not r['optimal'] or abs(r['objective']-1)>1e-7:
    raise RuntimeError('Model setup was skipped under python -O')
"""
        subprocess.run([sys.executable,'-O','-c',program],cwd=ROOT,check=True,
                       capture_output=True,text=True,timeout=30)


class IndependentVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bakery=lab1.load_data();cls.network=lab2.load_data()
        cls.x=np.array(cls.bakery['min_batches'],float)
        cls.b=np.array(cls.bakery['oven_hours'],float)
        cls.profit=float(np.array(cls.bakery['margin_rub_per_batch'])@cls.x)
        cls.flow=np.zeros((4,5,2))
        with (ROOT/'lab2/results/base_solution.csv').open(encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                i=int(row['warehouse'][1:])-1;j=int(row['customer'][1:])-1
                m=cls.network['modes'].index(row['mode'])
                cls.flow[i,j,m]=float(row['flow_t'])
        cls.injection=cls.flow.sum(axis=(1,2))
        cls.demand=np.array(cls.network['demand_t'],float)
        cls.cap=np.array(cls.network['capacity_t'],float)
        cls.cost=float(np.sum(np.array(cls.network['cost_rub_per_t'])*cls.flow))

    def test_valid_bakery_plan(self):
        self.assertLessEqual(lab1.verify(self.bakery,self.x,self.b,self.profit)['max_violation'],1e-7)

    def test_bakery_rejects_negative_nonfinite_and_wrong_objective(self):
        for value in [-.01,np.nan]:
            x=self.x.copy();x[-1]=value
            with self.subTest(value=value),self.assertRaises(ValueError):
                lab1.verify(self.bakery,x,self.b,self.profit)
        with self.assertRaises(ValueError):
            lab1.verify(self.bakery,self.x,self.b,self.profit+1)

    def test_network_checks_actual_technical_arcs(self):
        result=lab2.verify(self.network,self.flow,self.injection,self.demand,self.cap,self.cost,self.demand)
        self.assertEqual(result['max_violation'],0)
        delivery=self.demand.copy();delivery[0]+=1
        with self.assertRaises(ValueError):
            lab2.verify(self.network,self.flow,self.injection,self.demand,self.cap,self.cost,delivery)
        injection=self.injection.copy();injection[0]+=1;injection[1]-=1
        with self.assertRaises(ValueError):
            lab2.verify(self.network,self.flow,injection,self.demand,self.cap,self.cost,self.demand)

    def test_network_rejects_capacity_violation_and_nan(self):
        cap=self.cap.copy();cap[3,4,1]=0
        with self.assertRaises(ValueError):
            lab2.verify(self.network,self.flow,self.injection,self.demand,cap,self.cost)
        flow=self.flow.copy();flow[0,0,0]=np.nan
        with self.assertRaises(ValueError):
            lab2.verify(self.network,flow,self.injection,self.demand,self.cap,self.cost)


class DeliverableTests(unittest.TestCase):
    def test_variant_and_optimal_objectives(self):
        for i,expected in [(1,44783),(2,120651)]:
            with self.subTest(lab=i):
                d=json.loads((ROOT/f'lab{i}/data/variant.json').read_text())
                self.assertEqual(1+((17*d['N']+7*d['Q']+d['Y'])%20),17)
                self.assertEqual(10000+100*d['Q']+d['N'],10119)
                s=json.loads((ROOT/f'lab{i}/results/summary.json').read_text())
                self.assertAlmostEqual(s['scenarios']['base']['objective'],expected)
                self.assertTrue((ROOT/f'lab{i}/report.pdf').read_bytes().startswith(b'%PDF'))

    def test_markdown_tables_have_contiguous_rows(self):
        for i in [1,2]:
            lines=(ROOT/f'lab{i}/report.md').read_text().splitlines()
            headers=0
            for n,line in enumerate(lines):
                if line.startswith('| ---'):
                    headers+=1
                    self.assertTrue(lines[n-1].startswith('|'))
                    self.assertTrue(lines[n+1].startswith('|'))
            self.assertGreater(headers,0)


if __name__=='__main__':
    unittest.main()
