"""Exactly 155 fixed new head fits, durable reservations and no retries."""
import time
import numpy as np
from safety import MANIFEST, dump, reference_path as ref, resources
from engine import read,load_state,event,array_sha
from science.scaling import heads
from science.stability.campaign import config
from science.export_v6.export import prepare_training
from science.crosssubject_aux_20261028 import train as cross,core as crosscore
from science.mean_aux_20261029 import train as mean,core as meancore

class Reservations:
    def __init__(self,output):
        self.output=output;self.path=output/'fit_reservations.json';self.rows=[]
        if self.path.exists():raise ValueError('no retry or resumption of a fit reservation')
    def fit(self,scope,name,call):
        caps=dict(semantic=50,export=5,cross=26,mean=74)
        if len(self.rows)>=155 or sum(reservation['scope']==scope for reservation in self.rows)>=caps[scope] or any(reservation['scope']==scope and reservation['id']==name for reservation in self.rows):raise ValueError('fixed fit allowance exceeded or duplicate')
        resources(self.output)
        row=dict(scope=scope,id=name,status='RESERVED',started_unix=time.time());self.rows.append(row);dump(self.path,self.rows)
        event('fit_reserved',scope=scope,id=name,count=len(self.rows))
        try:
            value=call();row.update(status='COMPLETE',seconds=time.time()-row['started_unix']);dump(self.path,self.rows);return value
        except BaseException as error:
            row.update(status='FAILED_OR_UNCERTAIN',error=repr(error));dump(self.path,self.rows);raise

def save_fit(folder,name,state,initial,receipt):
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/(name+'.npz'),**{'parameter__'+parameter_name:parameter_value for parameter_name,parameter_value in state.items()},**{'initial__'+parameter_name:parameter_value for parameter_name,parameter_value in initial.items()})
    dump(folder/(name+'.json'),receipt)

def train_all(catalog,output):
    data=catalog.data;reserve=Reservations(output);sem={};exp={};aux={};began=time.monotonic()
    def fixed(scope,name,records,matrix,excluded,seed):
        def execute():
            result=heads.fit_head(records,matrix,excluded,config(seed))
            save_fit(output/'fresh_fits'/scope,name,result['state'],result['initial_state'],result['evidence'])
            return result['state']
        return reserve.fit(scope,name,execute)
    for benchmark in data.benchmarks:
        for seed in range(5):
            if time.monotonic()-began>=1800:raise TimeoutError('semantic legacy 1800-second scope ceiling')
            sem[benchmark,seed]=fixed('semantic',benchmark+'__'+str(seed),data.records,data.matrix,{benchmark},seed)
    augmented,features,excluded=prepare_training(data.records,data.matrix)
    for seed in range(5):exp[seed]=fixed('export','seed'+str(seed),augmented,features,excluded,seed)
    del features,augmented
    controls={};poison_results={};all_differences={}
    for scope,module,core in [('cross',cross,crosscore),('mean',mean,meancore)]:
        for benchmark in read(scope+'/protocol.json')['outers']:
            for seed in range(1 if scope=='cross' else 3):
                planned=core.schedule(data.records,catalog.inputs,{benchmark},seed)
                reference_name=f'{benchmark}__{seed}.npz' if scope=='mean' else f'{benchmark}__pooled__0.npz'
                with np.load(ref(f'{scope}/schedules/{reference_name}'),allow_pickle=False) as arrays:
                    if set(arrays.files)!=set(planned) or any(not np.array_equal(arrays[parameter_name],parameter_value) for parameter_name,parameter_value in planned.items()):raise ValueError('frozen schedule differs before reservation')
                if scope=='mean':
                    targets=core.mean_targets(data.records,catalog.inputs,{benchmark})
                    expected=MANIFEST['mean_target_hashes'][benchmark]
                    if list(targets.shape)!=expected['shape'] or str(targets.dtype)!=expected['dtype'] or array_sha(targets)!=expected['sha256']:
                        raise ValueError('frozen mean targets differ')
                for arm in core.ARMS:
                    name=f'{benchmark}__{arm}__{seed}'
                    def execute(benchmark=benchmark,seed=seed,arm=arm,name=name):
                        model,initial,receipt=module.fit(data.records,data.matrix,catalog.inputs,{benchmark},arm,seed,expected_schedule=planned,resource_check=lambda:resources(output))
                        state=heads._state(model);controls[scope,name]=receipt
                        original=load_state(ref(f'{scope}/models/{name}.npz'))
                        receipt['reference_parameter_max_abs_error']=max(float(np.max(np.abs(state[parameter_name].astype(np.float64)-original[parameter_name]))) for parameter_name in state)
                        if arm=='pooled':
                            pooled=sem[benchmark,seed]
                            exact=all(np.array_equal(state[parameter_name],pooled[parameter_name]) for parameter_name in state)
                            receipt['fresh_semantic_pooled_state_exact']=exact
                            if not exact:raise ValueError('pooled control differs from same new semantic fixed seed')
                        save_fit(output/'fresh_fits'/scope,name,state,initial,receipt)
                        all_differences[scope+'/'+name]=receipt['reference_parameter_max_abs_error']
                        return state
                    aux[scope,name]=reserve.fit(scope,name,execute)
        for arm in (('intact','shuffled') if scope=='cross' else ('mean','shuffled')):
            benchmark='algotune';seed=0;normal=f'{benchmark}__{arm}__0';name=normal+'__poison'
            def execute_poison(benchmark=benchmark,arm=arm,normal=normal,name=name):
                records,matrix,inputs,changed=module.poisoned(data.records,data.matrix,catalog.inputs,benchmark)
                planned=core.schedule(data.records,catalog.inputs,{benchmark},0)
                model,initial,receipt=module.fit(records,matrix,inputs,{benchmark},arm,0,expected_schedule=planned,resource_check=lambda:resources(output))
                state=heads._state(model);original=aux[scope,normal]
                exact=all(np.array_equal(state[parameter_name],original[parameter_name]) for parameter_name in state)
                prior_receipt=controls[scope,normal]
                fields=('training','schedule_hashes','losses','main_losses','auxiliary_losses','preclip_gradient_norms')
                path_exact=all(receipt[parameter_name]==prior_receipt[parameter_name] for parameter_name in fields)
                if not exact or not path_exact:raise ValueError('excluded raw-source poison changed actual fitted state or loss/gradient path')
                receipt.update(actual_refit=True,excluded_records_changed=len(changed),state_exact=exact,loss_gradient_path_exact=path_exact)
                save_fit(output/'fresh_fits'/scope,name,state,initial,receipt)
                poison_results[scope+'/'+arm]=dict(actual_refit=True,state_exact=exact,loss_gradient_path_exact=path_exact,changed_records=len(changed))
            reserve.fit(scope,name,execute_poison)
    if len(reserve.rows)!=155 or any(reservation['status']!='COMPLETE' for reservation in reserve.rows):raise ValueError('incomplete fixed fresh fit route')
    dump(output/'training_summary.json',dict(fits=155,semantic=50,export=5,cross=26,mean=74,poison_controls=poison_results,reference_parameter_max_abs_errors=all_differences,encoder_fit=False,seed_selection=False,retries=0))
    return sem,exp,aux
