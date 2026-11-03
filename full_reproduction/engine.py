"""Independent adapters around fixed original scientific functions."""
import hashlib
import importlib.metadata
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pyarrow as arrow
import pyarrow.parquet as pq
import torch
from scipy.special import expit
from safety import MANIFEST, PACKAGE, dump, reference_path as ref, resources, sha
from science.autopipeline.experiments import FIELDS
from science.autopipeline.semantic import feature_matrix, digest
from science.affine_calibration_20261015.core import SUBJECT_FIELDS, visible_key
from science.strategy.transfer import experiment as transfer
from science.head_tail_20261016.core import fuse_text_blocks
from science.scaling.heads import _Head
from science.stability.campaign import config

DATA = None

def setup():
    return DATA

def read(name):
    return json.loads(ref(name).read_text())

def array_sha(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()

def event(stage, **kw):
    print(json.dumps(dict(stage=stage, **kw),separators=(',',':')),flush=True)

def label_hash(records, excluded):
    return hashlib.sha256(json.dumps([[index,record['benchmark'],record['pair_id'],int(record['successes']),int(record['trials'])] for index,record in enumerate(records) if record['benchmark'] not in excluded],separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def split_projection(value):
    if isinstance(value,list):return [split_projection(x) for x in value]
    if isinstance(value,dict):
        return {k+'_sha256' if k in ('acquired_labels','acquired_sums') else k:
                hashlib.sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest() if k in ('acquired_labels','acquired_sums') else split_projection(v)
                for k,v in value.items()}
    return value

class Data:
    def __init__(self, data_root, output, encoder=None):
        transfer.DATA_ROOT=data_root
        self.pairs, exclusions, audits, hashes=transfer.load_data(dict(benchmarks=MANIFEST['selection_order'],revision=MANIFEST['raw_revision']))
        if hashes!={key:metadata['sha256'] for key,metadata in MANIFEST['raw_files'].items()}:raise ValueError('raw identity drift')
        self.benchmarks=sorted({pair.benchmark for pair in self.pairs})
        self.splits={(pair.pair_id,seed):transfer.split_pair(pair,seed) for pair in self.pairs for seed in range(5)}
        split_bytes=''.join(json.dumps(split_projection(input_value),sort_keys=True,separators=(',',':'))+'\n' for input_value in self.splits.values()).encode()
        if split_bytes!=ref('splits.jsonl').read_bytes():raise ValueError('735 reconstructed splits differ')
        self.evidence=dict(heldout_benchmarks=self.benchmarks,raw_hashes=hashes,confirmation_reserve_untouched=True,exclusions=exclusions,eligibility_audits=audits)
        self.records=[]
        for benchmark in self.benchmarks:
            im={record['item_id']:record['content'] for record in pq.read_table(data_root/benchmark/'items.parquet',columns=['item_id','content']).to_pylist()}
            sm={record['subject_id']:record for record in pq.read_table(data_root/benchmark/'subjects.parquet',columns=['subject_id',*FIELDS]).to_pylist()}
            for pair in self.pairs:
                if pair.benchmark!=benchmark:continue
                for item, trials in sorted(pair.by_item.items()):
                    sub={field:'' if sm[pair.subject_id].get(field) is None or isinstance(sm[pair.subject_id].get(field),float) and math.isnan(sm[pair.subject_id][field]) else str(sm[pair.subject_id][field]) for field in FIELDS}
                    self.records.append(dict(benchmark=benchmark,pair_id=pair.pair_id,item_id=item,content=im[item] if isinstance(im[item],str) else '',subject=sub,successes=sum(input_value for _,input_value in trials),trials=len(trials)))
        if (len(self.pairs),len(self.benchmarks),len(self.records),len(self.splits))!=(147,10,76455,735):raise ValueError('frozen data universe changed')
        self.texts=sorted({record['content'] for record in self.records});self.lookup={text:index for index,text in enumerate(self.texts)}
        hashes=[hashlib.sha256(text.encode()).hexdigest() for text in self.texts]
        if hashes!=read('text_hashes.json'):raise ValueError('original text order changed')
        self.embedding_key=read('embedding_manifest.json')['key']
        if read('embedding_manifest.json')['identity']['text_sha256']!=hashes:raise ValueError('embedding order mismatch')
        with np.load(ref('embeddings.npz'),allow_pickle=False) as arrays:embeddings=arrays['vectors'].copy()
        with np.load(ref('tail_vectors.npz'),allow_pickle=False) as arrays:
            self.affected=arrays['affected_indices'].copy();tails=arrays['vectors'].copy()
            if not arrays['completed'].all():raise ValueError('tail cache incomplete')
        if encoder:
            from encoding import rebuild
            embeddings,self.affected,tails=rebuild(self.texts,encoder,output,embeddings,self.affected,tails)
            self.embedding_key='fresh-'+array_sha(embeddings)
        if embeddings.shape!=(len(self.texts),384) or embeddings.dtype!=np.float32 or not np.isfinite(embeddings).all() or not np.allclose(np.linalg.norm(embeddings,axis=1),1,atol=2e-4):raise ValueError('invalid embeddings')
        self.matrix=feature_matrix(self.records,embeddings,self.lookup)
        self.record_identity=digest([dict(benchmark=record['benchmark'],pair_id=record['pair_id'],item_id=record['item_id'],text=hashlib.sha256(record['content'].encode()).hexdigest(),subject=record['subject']) for record in self.records])
        by_text=np.zeros((len(self.texts),384),np.float32);mask=np.zeros(len(self.texts),np.bool_)
        by_text[self.affected]=tails;mask[self.affected]=True
        order=np.asarray([self.lookup[record['content']] for record in self.records]);self.record_mask=mask[order]
        self.inference=self.matrix.copy();self.inference[:,:384]=fuse_text_blocks(self.matrix[:,:384],by_text[order],self.record_mask)
        dump(output/'data_identity.json',dict(records=len(self.records),pairs=len(self.pairs),splits=len(self.splits),benchmarks=self.benchmarks,record_identity=self.record_identity,embedding_key=self.embedding_key,feature_sha256=array_sha(self.matrix),headtail_feature_sha256=array_sha(self.inference),raw_audits=audits,confirmation_labels_used=False))
        self.data_root=data_root
    def indices(self,excluded):
        return ([index for index,record in enumerate(self.records) if record['benchmark'] not in excluded],[index for index,record in enumerate(self.records) if record['benchmark'] in excluded])
    def map_predictions(self,target,prediction):
        if len(target)!=len(prediction) or not np.isfinite(prediction).all() or np.any((prediction<0)|(prediction>1)):raise ValueError('invalid mapped probabilities')
        out=defaultdict(dict)
        for index,probability in zip(target,prediction):out[self.records[index]['pair_id']][self.records[index]['item_id']]=float(probability)
        return dict(out)

class coordinator:
    def __init__(self,data):
        self.data=data;self.inputs=[];self.index={}
        aliases={benchmark:f'benchmark_{index:03d}' for index,benchmark in enumerate(data.benchmarks)};items={}
        for benchmark in data.benchmarks:
            folder=data.data_root/benchmark
            for input_value in pq.read_table(folder/'responses.parquet',columns=['interactors']).column(0).to_pylist():
                if (input_value or '').strip():raise ValueError('response-specific interactors unsupported by original contract')
            for record in pq.read_table(folder/'items.parquet',columns=['item_id','content','item_features']).to_pylist():items[benchmark,record['item_id']]=dict(item_content=record['content'] or '',item_features=(record['item_features'] or '').strip(),interactors='',benchmark_id=aliases[benchmark])
        owners={}
        for record_index,record in enumerate(data.records):
            sub={field:(record['subject'].get(field) or '').strip() for field in SUBJECT_FIELDS}
            if sub!=record['subject']:raise ValueError('subject normalization differs')
            input_value=[sub,items[record['benchmark'],record['item_id']]];key=visible_key(input_value)
            if key in owners and not np.allclose(data.matrix[record_index],data.matrix[owners[key]],rtol=0,atol=1e-7):raise ValueError('equal visible features differ')
            owners.setdefault(key,record_index);self.inputs.append(input_value);self.index[record['pair_id'],record['item_id']]=record_index
        self.pairs={pair.pair_id:pair for pair in data.pairs}
        self.splits={(pair.pair_id,seed):ceil_split(pair,seed) for pair in data.pairs for seed in range(5)}
        if split_projection(list(self.splits.values()))!=read('ceil_splits.json')['ceil']:raise ValueError('reconstructed ceil splits differ')
        self.cost={pid:len(self.splits[pid,0]['acquisition_items'])+len(self.splits[pid,0]['evaluation_items']) for pid in self.pairs};self.panels={}

    def panel(self, target, seed, context, outer=None):
            cache_key = (target, seed, context, outer)
            if cache_key in self.panels:
                return self.panels[cache_key]
            pairs = sorted(pid for pid, pair in self.pairs.items() if pair.benchmark != outer)
            if target not in pairs:
                raise ValueError('outer target forbidden in inner context')
            if context == 'dense147':
                selected = pairs
            elif self.cost[target] > 1000:
                selected = None
            else:
                benchmark = self.pairs[target].benchmark
                candidates = [pid for pid in pairs if pid != target and (context == 'restricted_mixed_balanced' or self.pairs[pid].benchmark == benchmark)]
                remaining = 1000-self.cost[target]
                selected = [target]
                counts = Counter({benchmark: 1})
                while candidates:
                    candidates.sort(key=lambda pid: (counts[self.pairs[pid].benchmark] if context == 'restricted_mixed_balanced' else 0, panel_digest([seed, target, pid, context])))
                    donor = candidates.pop(0)
                    if self.cost[donor] <= remaining:
                        selected.append(donor)
                        remaining -= self.cost[donor]
                        counts[self.pairs[donor].benchmark] += 1
            self.panels[cache_key] = selected
            return selected

def ceil_split(pair, seed):
    ids = sorted(pair.by_item)
    seeded(seed, pair.pair_id, 'split').shuffle(ids)
    cut = (len(ids)+1)//2
    acquisition, evaluation = ids[:cut], sorted(ids[cut:])
    order = sorted(acquisition)
    generator = seeded(seed, pair.pair_id, 'acquisition')
    generator.shuffle(order)
    chosen = [generator.choice(sorted(pair.by_item[item])) for item in order[:31]]
    return dict(pair_id=pair.pair_id, benchmark=pair.benchmark, seed=seed,
        acquisition_items=sorted(acquisition), evaluation_items=evaluation, acquired_items=order[:31],
        selected_response_ids=[r[0] for r in chosen], acquired_labels=[r[1] for r in chosen])

seeded=transfer.seeded
def panel_digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def load_state(path, prefix='parameter__'):
    with np.load(path,allow_pickle=False) as arrays:
        state={parameter_name[len(prefix):]:arrays[parameter_name].copy() for parameter_name in arrays.files if parameter_name.startswith(prefix)} if prefix else {parameter_name:arrays[parameter_name].copy() for parameter_name in arrays.files}
    shapes={'linear_weight':(512,),'linear_bias':(),'text_factor':(384,8),'metadata_factor':(128,8)}
    if set(state)!=set(shapes) or any(state[parameter_name].shape!=shape or state[parameter_name].dtype!=np.float32 or not np.isfinite(state[parameter_name]).all() for parameter_name,shape in shapes.items()):raise ValueError('invalid saved bilinear head')
    return state

def forward(state, features, seed=0):
    model=_Head(config(seed),torch.Generator(device='cpu').manual_seed(0))
    model.load_state_dict({parameter_name:torch.from_numpy(parameter_value) for parameter_name,parameter_value in state.items()});model.eval()
    values=[]
    with torch.inference_mode():
        for start in range(0,len(features),4096):values.append(torch.sigmoid(model(torch.from_numpy(features[start:start+4096].copy()))).numpy().copy())
    return np.concatenate(values)

def compare_tree(actual,expected):
    if isinstance(expected,dict):
        if set(actual)!=set(expected):raise ValueError('aggregate keys differ')
        return max((compare_tree(actual[key],expected_entry) for key,expected_entry in expected.items()),default=0.)
    if isinstance(expected,list):
        if len(actual)!=len(expected):raise ValueError('aggregate lengths differ')
        return max((compare_tree(actual_entry,expected_entry) for actual_entry,expected_entry in zip(actual,expected)),default=0.)
    if isinstance(expected,(float,int)) and not isinstance(expected,bool):return abs(float(actual)-float(expected))
    if actual!=expected:raise ValueError('non-numeric aggregate/gate differs: '+repr((actual,expected)))
    return 0.

def semantic(data,out,states=None,strict=True):
    from science.online_acquisition_20261015.data import VisiblePrior
    from science.stability.campaign import score,compose_rows,MODES
    from science.scaling.head_campaign import aggregate_rows
    from science.head_tail_20261016.campaign import numpy_forward
    priors={seed:{} for seed in range(5)};ensemble={};headtail={};cached_error=0.;independent_error=0.;headtail_error=0.
    with np.load(ref('headtail_priors.npz'),allow_pickle=False) as arrays:anchor={benchmark:arrays[benchmark].copy() for benchmark in data.benchmarks}
    for benchmark in data.benchmarks:
        target=data.indices({benchmark})[1];predictions=[];fused=[]
        for seed in range(5):
            evidence=read(f'semantic/{benchmark}/{seed}/evidence.json')
            if evidence['excluded']!=[benchmark] or evidence['training_label_hash']!=label_hash(data.records,{benchmark}) or evidence['record_identity']!=data.record_identity:raise ValueError('original head source exclusion/record mismatch')
            if strict and evidence['embedding_key']!=data.embedding_key:raise ValueError('original embedding key mismatch')
            state=states[benchmark,seed] if states is not None else load_state(ref(f'semantic/{benchmark}/{seed}/model.npz'))
            prior_probability=forward(state,data.matrix[target],seed)
            with np.load(ref(f'semantic/{benchmark}/{seed}/model.npz'),allow_pickle=False) as arrays:
                if not np.array_equal(arrays['target_indices'],target):raise ValueError('target indices differ')
                cached_error=max(cached_error,float(np.max(np.abs(prior_probability.astype(np.float64)-arrays['predictions']))))
            predictions.append(prior_probability.astype(np.float64));priors[seed].update(data.map_predictions(target,prior_probability))
            fp=forward(state,data.inference,seed);fused.append(fp.astype(np.float64))
            independent_error=max(independent_error,float(np.max(np.abs(numpy_forward(data.inference,state).astype(np.float64)-fp.astype(np.float64)))))
        ensemble.update(data.map_predictions(target,np.mean(predictions,axis=0)))
        callback=VisiblePrior(DATA_CATALOG.inputs,np.mean(fused,axis=0));canonical=np.asarray([callback(entry) for entry in DATA_CATALOG.inputs],np.float64)
        headtail[benchmark]=canonical;headtail_error=max(headtail_error,float(np.max(np.abs(canonical-anchor[benchmark]))))
        event('semantic_forward',source=benchmark,heads=5,cached_max_abs_error=cached_error,headtail_max_abs_error=headtail_error)
        resources(out)
    baseline={};linear_error=0.
    for benchmark in data.benchmarks:
        target=data.indices({benchmark})[1]
        with np.load(ref(f'linear/{benchmark}.npz'),allow_pickle=False) as arrays:
            if not np.array_equal(arrays['target_indices'],target):raise ValueError('linear target mismatch')
            prior_probability=expit(data.matrix[target].astype(np.float64)@arrays['coef']+float(arrays['intercept'][0]));linear_error=max(linear_error,float(np.max(np.abs(prior_probability-arrays['predictions']))))
        baseline.update(data.map_predictions(target,prior_probability))
    rows={'map_reference':score(data,baseline,'map_reference',strength=2.)}
    rows.update({f'training_seed{seed}':score(data,priors[seed],f'training_seed{seed}') for seed in range(5)})
    learned=score(data,ensemble,'ensemble')
    rows.update({'ensemble_'+mean_values:compose_rows(learned,rows['map_reference'],mean_values) for mean_values in MODES})
    score_error=0.
    for name,values in rows.items():
        original=read('stability_scores/'+name+'.json')
        if len(values)!=len(original):raise ValueError('stability score cardinality differs')
        original_by_key={(record['pair_id'],record['seed'],record['budget']):record for record in original}
        if len(original_by_key)!=len(original):raise ValueError('duplicate original scoring cell')
        for actual_row in values:
            expected_row=original_by_key[actual_row['pair_id'],actual_row['seed'],actual_row['budget']]
            if {key:value for key,value in actual_row.items() if key!='brier'}!={key:value for key,value in expected_row.items() if key!='brier'}:raise ValueError('stability score metadata differs')
            score_error=max(score_error,abs(actual_row['brier']-expected_row['brier']))
        dump(out/'semantic'/('scores_'+name+'.json'),values)
    summaries={name:aggregate_rows(values) for name,values in rows.items()}
    summary_error=compare_tree(summaries,read('stability_report.json')['results']['summaries'])
    result=dict(implemented=True,executed=True,passed=bool(score_error<=2e-12 and summary_error<=2e-12 and cached_error<=1e-7 and headtail_error<=1e-7),heads_forwarded=50,score_rows=sum(map(len,rows.values())),cached_forward_max_abs_error=cached_error,linear_forward_max_abs_error=linear_error,headtail_forward_max_abs_error=headtail_error,numpy_torch_max_abs_error=independent_error,score_max_abs_error=score_error,aggregate_max_abs_error=summary_error,summaries=summaries,heldout_policy=MANIFEST['heldout_policy'])
    dump(out/'semantic/summary.json',result)
    if strict and not result['passed']:raise ValueError('semantic replay parity failure')
    return headtail,result

def export_forward(data,out,states=None):
    from science.head_tail_20261016.campaign import numpy_forward
    folder=out/'full_export';folder.mkdir(exist_ok=True);probability=[];error=0.;parameter_error=0.;reference_error=0.
    for seed in range(5):
        state=states[seed] if states is not None else load_state(ref(f'export/{seed}.npz'),prefix='')
        np.savez_compressed(folder/f'head_seed{seed}.npz',**state)
        values=forward(state,data.matrix,seed);probability.append(values.astype(np.float64))
        error=max(error,float(np.max(np.abs(values.astype(np.float64)-numpy_forward(data.matrix,state).astype(np.float64)))))
        original=load_state(ref(f'export/{seed}.npz'),prefix='')
        parameter_error=max(parameter_error,max(float(np.max(np.abs(state[key].astype(np.float64)-original[key]))) for key in state))
        reference_error=max(reference_error,float(np.max(np.abs(values.astype(np.float64)-forward(original,data.matrix,seed).astype(np.float64)))))
    mean=np.mean(probability,axis=0)
    losses=[record['successes']*(1-value)**2+(record['trials']-record['successes'])*value*value for record,value in zip(data.records,mean)]
    result=dict(implemented=True,executed=True,passed=bool(error<=2e-6 and reference_error<=1e-7),heads=5,predictions=len(mean),mean_prior_sha256=array_sha(mean),numpy_torch_max_abs_error=error,reference_parameter_max_abs_error=parameter_error,reference_forward_max_abs_error=reference_error,training_brier=float(math.fsum(losses)/sum(record['trials'] for record in data.records)),heldout=False,deployment_compatibility='unexecuted; local fixed-feature head export only')
    np.savez_compressed(folder/'forward_predictions.npz',probabilities=mean)
    dump(folder/'summary.json',result)
    return result

DATA_CATALOG=None

def transfer_forward(catalog,headtail,out,states=None):
    from science.online_acquisition_20261015.data import VisiblePrior
    result={};maximum=0.;ensemble_error=0.;poison_checks={}
    for scope in ['cross','mean']:
        arms=('intact','shuffled','pooled') if scope=='cross' else ('mean','shuffled','pooled')
        with np.load(ref(scope+'_priors.npz'),allow_pickle=False) as arrays:anchors={key:arrays[key].copy() for key in arrays.files}
        with np.load(ref('mean_ensemble_priors.npz'),allow_pickle=False) as arrays:ensemble_anchors={key:arrays[key].copy() for key in arrays.files} if scope=='mean' else {}
        result[scope]={}
        for benchmark in read(scope+'/protocol.json')['outers']:
            callbacks={'old_full':VisiblePrior(catalog.inputs,headtail[benchmark])}
            for arm in arms:
                raw=[]
                for seed in range(1 if scope=='cross' else 3):
                    name=f'{benchmark}__{arm}__{seed}';state=states[scope,name] if states else load_state(ref(f'{scope}/models/{name}.npz'))
                    raw_prior=forward(state,catalog.data.inference,seed).astype(np.float64);raw.append(raw_prior)
                    key=benchmark+'__'+arm if scope=='cross' else name
                    maximum=max(maximum,float(np.max(np.abs(raw_prior-anchors[key]))))
                    callbacks[arm if scope=='cross' else arm+'_s'+str(seed)]=VisiblePrior(catalog.inputs,raw_prior)
                if scope=='mean':
                    mean_prior=np.mean(raw,axis=0);ensemble_error=max(ensemble_error,float(np.max(np.abs(mean_prior-ensemble_anchors[benchmark+'__'+arm]))))
                    callbacks[arm+'_ensemble']=VisiblePrior(catalog.inputs,mean_prior)
            result[scope][benchmark]=callbacks
            event('transfer_forward',scope=scope,source=benchmark,raw_prior_max_abs_error=maximum)
        for arm in (('intact','shuffled') if scope=='cross' else ('mean','shuffled')):
            name='algotune__'+arm+'__0'
            normal=load_state(ref(scope+'/models/'+name+'.npz'))
            poisoned=load_state(ref(scope+'/models/'+name+'__poison.npz'))
            exact=all(np.array_equal(normal[key],poisoned[key]) for key in normal)
            if not exact:raise ValueError('saved historical poison control state differs')
            poison_checks[scope+'/'+arm]=dict(saved_states_exact=exact,new_poison_refit=False)
        resources(out)
    return result,dict(heads_forwarded=96,raw_prior_max_abs_error=maximum,ensemble_prior_max_abs_error=ensemble_error,historical_poison_state_checks=poison_checks)

def score_transfer(catalog,headtail,callbacks,out,strict=True):
    import gzip
    from collections import OrderedDict
    from science.online_acquisition_20261015.data import VisiblePrior
    from science.online_acquisition_20261015.stream import CAPS,Replay,visible_group
    from science.proxy_factorial_20261028 import campaign as proxy
    from science.crosssubject_aux_20261028 import campaign as cross
    from science.mean_aux_20261029 import campaign as mean
    from science.demand_features_20261026.features import raw as extract_raw
    methods={'proxy':proxy.METHODS,'cross':cross.METHODS}
    text_by_hash={hashlib.sha256(t.encode()).hexdigest():t for t in catalog.data.texts}
    contents=[text_by_hash[r['sha256']] for r in read('contents.json')]
    if len(contents)!=len(set(contents)) or set(contents)!=set(catalog.data.texts):raise ValueError('proxy content coverage differs')
    with np.load(ref('features.npz'),allow_pickle=False) as arrays:raw=arrays['raw24'].copy();unit=arrays['unit25'].copy()
    rebuilt=np.stack([extract_raw(token) for token in contents])
    if not np.array_equal(rebuilt,raw):raise ValueError('24 original label-free features differ')
    with np.load(ref('permutation.npz'),allow_pickle=False) as arrays:permutation=arrays['permutation'].copy()
    if not np.array_equal(np.sort(permutation),np.arange(len(contents))):raise ValueError('invalid frozen shuffle')
    semantic={}
    for index,value in enumerate(catalog.inputs):
        content=visible_key(value)[8]
        if content not in semantic:
            vector=catalog.data.matrix[index,:384].astype(np.float64);semantic[content]=vector/np.linalg.norm(vector)
    demand=dict(zip(contents,unit));visible_groups=[visible_group(value) for value in catalog.inputs]
    priors={benchmark:VisiblePrior(catalog.inputs,headtail[benchmark]) for benchmark in catalog.data.benchmarks}
    evaluator=proxy.Evaluator(SimpleNamespace(catalog=catalog,priors=priors,semantic=semantic,raw_map=dict(zip(contents,raw)),shuffled_map=dict(zip(contents,raw[permutation])),visible_groups=visible_groups))
    anchors={};rows={seed:[] for seed in ('proxy','cross','mean')};checks={};cache=OrderedDict()
    for scope in rows:
        ar=pq.read_table(ref(scope+'/scores.parquet')).to_pylist()
        with np.load(ref(scope+'/predictions.npz'),allow_pickle=False) as arrays:
            anchors[scope]=(ar,arrays['probabilities'].copy(),arrays['offsets'].copy())
        checks[scope]=dict(prediction_max_abs_error=0.,score_max_abs_error=0.,metadata_exact=True,observations_exact=True)
    count=decisions=0;started=time.monotonic()
    with gzip.open(ref('traces.jsonl.gz'),'rt') as stream:
        for line in stream:
            trace=json.loads(line);target,seed,context=trace['target'],trace['seed'],trace['context']
            if (target,seed,context)!=(catalog.targets[count//15],(count//3)%5,proxy.CONTEXTS[count%3]):raise ValueError('trajectory order changed')
            benchmark=catalog.pairs[target].benchmark;panel=catalog.panel(target,seed,context);query=catalog.evaluation([target],seed)
            group=catalog.groups(panel,seed)
            if [group.name for group in group]!=trace['groups']:raise ValueError('frozen panel groups differ')
            if catalog.evaluation(panel,seed)[:len(query)]!=query:raise ValueError('target-first evaluation order differs')
            replay=Replay(group,lambda token:catalog.inputs[token],lambda token:catalog.responses[seed,token],priors[benchmark],'random')
            queries=[catalog.inputs[token] for token in query];success,trials=(np.asarray(entry,dtype=np.int32) for entry in catalog.outcomes(target,seed))
            own_group=visible_groups[query[0]]
            for point,cap in zip(trace['checkpoints'],CAPS):
                if point['cap']!=cap:raise ValueError('checkpoint caps differ')
                replay.advance(cap)
                if replay.barriers[-1]!=point['barrier']:raise ValueError('actual purchased observation barrier differs')
                actual_point=dict(labeled_tokens=list(replay.purchased),labeled_outcomes=[outcome for _,outcome in replay.labeled],query_tokens=query)
                for field,value in actual_point.items():
                    if field in point and point[field]!=value:raise ValueError('actual purchased observations differ')
                point=dict(point,**actual_point)
                own=[(token,outcome) for token,(_,outcome) in zip(replay.purchased,replay.labeled) if visible_groups[token]==own_group]
                signature=(benchmark,tuple(query),tuple(sorted(own)),context==proxy.CONTEXTS[0])
                if signature in cache:
                    values,own_count=cache[signature];cache.move_to_end(signature)
                else:
                    proxy_values,own_count=evaluator.evaluate(benchmark,point)
                    cross_values,states=cross.vectors(callbacks['cross'][benchmark],semantic,demand,queries,replay.labeled)
                    mean_values,mc=mean.vectors(callbacks['mean'][benchmark],semantic,demand,queries,replay.labeled,mean.methods(context))
                    if own_count!=mc or own_count!=next(iter(states.values()))[2]:raise ValueError('matched observation counts differ')
                    values=dict(proxy=proxy_values,cross=cross_values,mean=mean_values);cache[signature]=(values,own_count)
                    if len(cache)>512:cache.popitem(last=False)
                for scope in rows:
                    selected=mean.methods(context) if scope=='mean' else methods[scope]
                    for method in selected:
                        vector=values[scope][method]
                        loss=float(math.fsum(value*(1-prob)**2+(total-value)*prob*prob for prob,value,total in zip(vector,success,trials))/int(trials.sum()))
                        row=dict(context=context,target_pair=target,target_benchmark=benchmark,seed=seed,cap=cap,method=method,loss=loss,trial_count=int(trials.sum()),evaluation_items=len(trials),actual_target_labels=own_count,trajectory=count)
                        index=len(rows[scope]);old,prob,offset=anchors[scope];expected=old[index]
                        if {key:value for key,value in row.items() if key!='loss'}!={key:value for key,value in expected.items() if key!='loss'}:raise ValueError('score metadata differs: '+repr((scope,index)))
                        checks[scope]['score_max_abs_error']=max(checks[scope]['score_max_abs_error'],abs(loss-expected['loss']))
                        original=prob[offset[index]:offset[index+1]]
                        checks[scope]['prediction_max_abs_error']=max(checks[scope]['prediction_max_abs_error'],float(np.max(np.abs(vector-original))))
                        rows[scope].append(row)
            if replay.decisions!=trace['decisions'] or replay.purchased!=trace['purchased']:raise ValueError('actual acquisition decisions differ')
            count+=1;decisions+=len(replay.decisions)
            if count%50==0:
                dump(out/'progress.json',dict(stage='scoring',trajectories=count,total=1950,seconds=time.monotonic()-started,checks=checks,resources=resources(out)))
                event('scoring',trajectories=count,seconds=round(time.monotonic()-started,2))
    if (count,decisions)!=(1950,670165):raise ValueError('trace/decision universe changed')
    expected_count=dict(proxy=140400,cross=70200,mean=105300);modules=dict(proxy=proxy,cross=cross,mean=mean);results={}
    for scope in rows:
        if len(rows[scope])!=expected_count[scope] or len(rows[scope])!=len(anchors[scope][0]):raise ValueError('score universe differs')
        module=modules[scope];groups=module.aggregate(rows[scope]);gates=module.gates(groups)
        frozen_groups=read(scope+'/groups.json')
        if scope=='proxy':frozen_groups=frozen_groups['aggregates']
        if scope=='cross':frozen_groups=read(scope+'/summary.json')['aggregates']
        agg_error=compare_tree(groups,frozen_groups)
        original_summary=read(scope+'/summary.json');original_gates=original_summary['gates']
        gate_matches=True
        try:gate_error=compare_tree(gates,original_gates)
        except ValueError:
            if strict:raise
            gate_error=None;gate_matches=False
        negatives=[]
        primary='structure' if scope=='proxy' else 'intact' if scope=='cross' else 'mean_ensemble'
        blocks=defaultdict(dict)
        for row in rows[scope]:blocks[row['trajectory'],row['cap']][row['method']]=row
        for block in blocks.values():
            main=block[primary]
            for method,row in block.items():
                if method!=primary and main['loss']>=row['loss']-1e-12:negatives.append(dict(context=main['context'],target=main['target_pair'],seed=main['seed'],cap=main['cap'],candidate=primary,control=method,candidate_loss=main['loss'],control_loss=row['loss'],difference=main['loss']-row['loss']))
        folder=out/scope;folder.mkdir(exist_ok=True)
        pq.write_table(arrow.Table.from_pylist(rows[scope]),folder/'scores.parquet',compression='zstd')
        dump(folder/'groups.json',groups)
        with gzip.open(folder/'negative_groups.jsonl.gz','wt') as stream:
            for row in negatives:stream.write(json.dumps(row,separators=(',',':'))+'\n')
        group_negative=[]
        if scope=='proxy':
            contrasts,group_negative=module.contrast_groups(groups)
            dump(folder/'all_original_group_contrasts.json',contrasts)
            if strict and len(group_negative)!=original_summary['negative_group_count']:raise ValueError('original proxy negative contrast count differs')
        if scope=='mean':
            for context,group in groups.items():
                for level,collection in (('source',group['sources']),('target',group['targets']),('split',group['seeds']),('overall',{'all':group['mean']})):
                    for key,values in collection.items():
                        for control,value in values.items():
                            if control!=primary and values[primary]>=value-1e-12:group_negative.append(dict(level=level,context=context,key=key,control=control,primary_loss=values[primary],control_loss=value,gain=value-values[primary]))
        with gzip.open(folder/'original_negative_group_contrasts.jsonl.gz','wt') as stream:
            for row in group_negative:stream.write(json.dumps(row,separators=(',',':'))+'\n')
        results[scope]=dict(implemented=True,executed=True,passed=bool(checks[scope]['prediction_max_abs_error']<=2e-12 and checks[scope]['score_max_abs_error']<=2e-12 and agg_error<=2e-12 and gate_matches and gate_error<=2e-12),**checks[scope],aggregate_max_abs_error=agg_error,gate_max_abs_error=gate_error,gate_decisions_match=gate_matches,score_rows=len(rows[scope]),predictions_compared=int(anchors[scope][2][-1]),trajectories=count,decisions=decisions,negative_rows=len(negatives),original_negative_group_contrasts=len(group_negative),gates=gates,repeated_development=True,fresh_generalization=False)
        dump(folder/'summary.json',results[scope])
    if strict and not all(entry['passed'] for entry in results.values()):raise ValueError('transfer replay parity failure')
    return results

def run(mode,data_root,reference_root,output,encoder=None):
    global DATA,DATA_CATALOG
    torch.set_num_threads(2);arrow.set_cpu_count(2);arrow.set_io_thread_count(2)
    started=time.monotonic();DATA=Data(data_root,output,encoder)
    from science.online_acquisition_20261015.data import Catalog
    DATA_CATALOG=Catalog();event('data_ready',records=len(DATA.records),targets=len(DATA_CATALOG.targets))
    sem_states=exp_states=aux_states=None
    if mode=='train':
        from training import train_all
        sem_states,exp_states,aux_states=train_all(DATA_CATALOG,output)
    strict=mode=='replay'
    headtail,semantic_result=semantic(DATA,output,sem_states,strict)
    export_result=export_forward(DATA,output,exp_states)
    callbacks,forward_result=transfer_forward(DATA_CATALOG,headtail,output,aux_states)
    transfer_results=score_transfer(DATA_CATALOG,headtail,callbacks,output,strict)
    report=dict(mode=mode,validation_level='cache-backed recomputed forwards and observations' if strict else 'fresh fixed fits'+(' and fresh encoder' if encoder else ' with original embeddings'),implemented=True,executed=True,passed=all(scope_result['passed'] for scope_result in [semantic_result,export_result,*transfer_results.values()]),five_scopes=dict(semantic=semantic_result,full_export=export_result,**transfer_results),auxiliary_forward=forward_result,fit_count=155 if mode=='train' else 0,fresh_encoding_executed=encoder is not None,elapsed_seconds=time.monotonic()-started,versions={distribution_name:importlib.metadata.version(distribution_name) for distribution_name in ['numpy','scipy','torch','pandas','pyarrow','scikit-learn','transformers']},resources=resources(output),confirmation_labels_used=False,publication=False,commander_acceptance='not asserted',limitations=['fixed repeated public development only','no fresh-source generalization or independent scientific review','full-data export has no heldout evaluation','hosted deployment compatibility unexecuted'])
    dump(output/'replay_report.json',report);event('complete',elapsed_seconds=report['elapsed_seconds'],passed=report['passed'])
