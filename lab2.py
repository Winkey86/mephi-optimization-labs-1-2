"""Variant 17: parallel road/rail arcs in a minimum cost flow LP."""
import json
import time
from common import ROOT,TOL,np,plt,solve,meta,csv_write,save_json,finish_plot


def load_data():
    return json.loads((ROOT/'lab2/data/variant.json').read_text())


def build_model(d, demand, cap):
    # Nodes: source, four warehouses, five shops, sink. Parallel mode arcs retained.
    arcs=[(0,1+i,0,float(s),'source',i,-1,-1) for i,s in enumerate(d['supply_t'])]
    for i in range(4):
        for j in range(5):
            for m in range(2):
                arcs.append((1+i,5+j,float(d['cost_rub_per_t'][i][j][m]),float(cap[i,j,m]),
                             'transport',i,j,m))
    arcs += [(5+j,10,0,float(q),'sink',-1,j,-1) for j,q in enumerate(demand)]
    B=np.zeros((11,len(arcs)))
    for k,(u,v,*_) in enumerate(arcs): B[u,k]=-1;B[v,k]=1
    balance=np.zeros(11);balance[0]=-sum(demand);balance[10]=sum(demand)
    upper=np.array([a[3] for a in arcs]); lower=np.zeros(len(arcs));lower[-5:]=demand
    return arcs,B,balance,lower,upper


def decode(x):
    return np.asarray(x[4:44]).reshape(4,5,2)


def verify(d, flow, injection, demand, cap, objective, delivery=None):
    flow, injection, demand, cap = map(lambda a: np.asarray(a, float),
                                      [flow, injection, demand, cap])
    delivery = demand.copy() if delivery is None else np.asarray(delivery, float)
    if (flow.shape != (4,5,2) or cap.shape != (4,5,2) or injection.shape != (4,)
            or demand.shape != (5,) or delivery.shape != (5,)):
        raise ValueError('Unexpected network solution dimensions')
    if not all(np.isfinite(a).all() for a in [flow,injection,demand,cap,delivery]) or not np.isfinite(objective):
        raise ValueError('Network solution and objective must be finite')
    supply=np.array(d['supply_t']); cost=np.array(d['cost_rub_per_t'])
    sent=np.sum(flow,axis=(1,2)); received=np.sum(flow,axis=(0,2))
    balances=np.r_[sum(demand)-sum(injection),injection-sent,received-delivery,
                   sum(delivery)-sum(demand)]
    v=float(max(0,-np.min(flow),np.max(flow-cap),-np.min(injection),
                np.max(injection-supply),np.max(abs(delivery-demand)),np.max(abs(balances))))
    obj=float(np.sum(cost*flow)); err=abs(obj-objective)
    if v>TOL or err>TOL:
        raise ValueError(f'Independent network verification failed: violation={v}, objective_error={err}')
    return {'max_violation':v,'objective_recomputed':obj,'objective_error':err,
            'node_balance_residuals':balances,'warehouse_out_t':sent,
            'warehouse_unused_t':supply-sent,'customer_in_t':received,'tolerance':TOL}


def greedy(d,demand,cap):
    start=time.perf_counter()
    supply=np.array(d['supply_t'],float); remaining=np.array(demand,float).copy()
    cost=np.array(d['cost_rub_per_t']); flow=np.zeros((4,5,2))
    for flat in np.argsort(cost.ravel(),kind='stable'):
        i,j,m=np.unravel_index(flat,cost.shape)
        amount=min(supply[i],remaining[j],cap[i,j,m])
        flow[i,j,m]=amount;supply[i]-=amount;remaining[j]-=amount
    return flow,remaining,time.perf_counter()-start


def graph(d,flow,cap,path):
    fig,ax=plt.subplots(figsize=(10,7))
    # Mode layers split parallel arcs; every physical arc has a flow/capacity label.
    for m in range(2):
        x0=m*6
        for i in range(4):
            for j in range(5):
                f=float(flow[i,j,m]); u=float(cap[i,j,m]); yy=4.5-i*1.25;zz=5-j*1.25
                ax.annotate('',xy=(x0+3.7,zz),xytext=(x0+.3,yy),
                   arrowprops={'arrowstyle':'->','color':'#185c94' if f>TOL else '#d7dce0',
                               'lw':.5+1.8*f/max(u,1),'alpha':.75})
                # Fixed distinct positions avoid overlapping labels at crossings.
                t=.20+.145*j
                lx=x0+.3+3.4*t;ly=yy*(1-t)+zz*t
                ax.text(lx,ly,f'{f:g}/{u:g}',fontsize=6.6,ha='center',va='center',
                        bbox={'facecolor':'white','edgecolor':'none','pad':.4},zorder=4)
        for i in range(4):
            ax.text(x0,4.5-i*1.25,f'С{i+1}',ha='center',va='center',fontsize=11,
                    bbox={'boxstyle':'round,pad=.3','fc':'#e6edf3','ec':'#627586'})
        for j in range(5):
            ax.text(x0+4,5-j*1.25,f'М{j+1}',ha='center',va='center',fontsize=11,
                    bbox={'boxstyle':'round,pad=.3','fc':'#e6edf3','ec':'#627586'})
        ax.text(x0+2,5.65,d['modes'][m],ha='center',weight='bold',fontsize=13)
    ax.set_xlim(-.6,10.6);ax.set_ylim(-.7,6.1);ax.axis('off')
    ax.text(5,-.6,'Подпись дуги: поток / ёмкость, т/сутки. Узлы повторены по видам транспорта.',ha='center',fontsize=9)
    finish_plot(path)


def run():
    d=load_data();out=ROOT/'lab2/results';out.mkdir(parents=True,exist_ok=True)
    cap=np.array(d['capacity_t'],float);dem=np.array(d['demand_t'],float)
    arcs,B,b,lo,hi=build_model(d,dem,cap);c=np.array([a[2] for a in arcs])
    base=solve(c,B,b,b,lo,hi,log=out/'base.log');assert base['optimal']
    f=decode(base['x']);busiest=np.unravel_index(np.argmax(f),f.shape)
    saturated=np.argwhere((abs(f-cap)<=TOL)&(cap>0))
    critical=tuple(max(saturated,key=lambda t:-base['col_dual'][4+np.ravel_multi_index(tuple(t),f.shape)]))
    csv_write(out/'node_potentials.csv',({'node':node,'potential_rub_per_t':v} for node,v in
               zip(['Исток','С1','С2','С3','С4','М1','М2','М3','М4','М5','Сток'],base['row_dual'])))
    overloaded=cap.copy();overloaded[critical]*=.8
    failed=cap.copy();failed[busiest]=0
    scenarios={'base':(dem,cap),'overload':(dem,overloaded),'demand_growth':(dem*1.2,cap),
               'arc_failure':(dem,failed),'stress':(dem*1.2,failed)}
    results={};checks={};records=[];heuristics=[]
    for name,(dd,cc) in scenarios.items():
        aa,BB,bb,ll,uu=build_model(d,dd,cc)
        r=base if name=='base' else solve(c,BB,bb,bb,ll,uu,log=out/f'{name}.log')
        results[name]={**meta(r),'flow':decode(r['x']) if r['optimal'] else None}
        records.append({'scenario':name,**meta(r)})
        if r['optimal']:
            flow=decode(r['x'])
            checks[name]=verify(d,flow,r['x'][:4],dd,cc,r['objective'],r['x'][-5:])
            rows=[]
            for i in range(4):
                for j in range(5):
                    for m in range(2):
                        k=4+(i*5+j)*2+m
                        rows.append({'warehouse':f'С{i+1}','customer':f'М{j+1}',
                         'mode':d['modes'][m],'flow_t':flow[i,j,m],'capacity_t':cc[i,j,m],
                         'cost_rub_per_t':c[k],'total_rub':flow[i,j,m]*c[k],
                         'capacity_shadow_rub_per_t':r['col_dual'][k]})
            csv_write(out/f'{name}_solution.csv',rows)
            csv_write(out/f'{name}_balances.csv',({'node':node,'residual_t':v} for node,v in
                      zip(['Исток','С1','С2','С3','С4','М1','М2','М3','М4','М5','Сток'],
                      checks[name]['node_balance_residuals'])))
            gf,left,seconds=greedy(d,dd,cc);feasible=bool(np.max(left)<=TOL)
            gc=float(np.sum(np.array(d['cost_rub_per_t'])*gf))
            if feasible: verify(d,gf,np.sum(gf,axis=(1,2)),dd,cc,gc)
            heuristics.append({'scenario':name,'feasible':feasible,'unmet_demand_t':float(sum(left)),
               'greedy_cost_rub':gc if feasible else None,'lp_cost_rub':r['objective'],
               'gap_percent':100*(gc/r['objective']-1) if feasible else None,
               'greedy_seconds':seconds,'lp_seconds':r['seconds']})
            csv_write(out/f'{name}_greedy.csv',({'warehouse':i+1,'customer':j+1,'mode':d['modes'][m],
                     'flow_t':gf[i,j,m]} for i in range(4) for j in range(5) for m in range(2)))
    # Network integrality: integer capacities and balances imply an integral LP vertex.
    integer=solve(c,B,b,b,lo,hi,integer=np.arange(len(c)),log=out/'integer.log')
    assert integer['optimal'] and abs(integer['objective']-base['objective'])<=TOL
    checks['integer']=verify(d,decode(integer['x']),integer['x'][:4],dem,cap,integer['objective'],integer['x'][-5:])
    checks['integer']['max_integrality_error']=float(np.max(abs(integer['x']-np.round(integer['x']))))
    assert checks['integer']['max_integrality_error']<=TOL
    checks['lp_integrality_error']=float(np.max(abs(base['x']-np.round(base['x']))))
    csv_write(out/'integer_solution.csv',({'arc':k,'flow_t':v} for k,v in enumerate(integer['x'])))
    k=4+np.ravel_multi_index(critical,f.shape);delta=.01
    capplus=cap.copy();capplus[critical]+=delta
    aa,BB,bb,ll,uu=build_model(d,dem,capplus)
    price=solve(c,BB,bb,bb,ll,uu,log=out/'capacity_plus_001.log')
    shadow={'arc':[int(v)+1 for v in critical], 'delta_t':delta,
            'shadow_rub_per_t':float(base['col_dual'][k]),
            'forecast_delta_rub':float(base['col_dual'][k]*delta),
            'actual_delta_rub':price['objective']-base['objective']}
    graph(d,f,cap,out/'network_base.png')
    graph(d,results['arc_failure']['flow'],failed,out/'network_failure.png')
    fig,ax=plt.subplots(figsize=(8,3.3));names=list(results)
    ax.bar(names,[results[n]['objective'] for n in names],color='#185c94')
    ax.set_ylabel('Стоимость, руб./сутки');ax.tick_params(axis='x',rotation=15)
    finish_plot(out/'scenarios.png')
    summary={'scenarios':results,'busiest_arc':[int(v)+1 for v in busiest],
             'busiest_flow_t':float(f[busiest]),'capacity_shadow_test':shadow,
             'node_potentials':base['row_dual'],
             'greedy':heuristics,'integer':meta(integer),'node_count':11,'arc_count':49,
             'physical_arc_count':40}
    save_json(out/'summary.json',summary);save_json(out/'verification.json',checks)
    csv_write(out/'scenarios.csv',records);csv_write(out/'greedy_comparison.csv',heuristics)
    return summary


if __name__=='__main__':
    print(run()['greedy'])
