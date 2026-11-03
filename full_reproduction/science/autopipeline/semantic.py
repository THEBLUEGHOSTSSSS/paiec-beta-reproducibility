from __future__ import annotations
import hashlib,json,math,statistics,itertools
from pathlib import Path
from collections import defaultdict,OrderedDict,Counter
import numpy as np
from scipy.special import expit,logit
BUDGETS=[0,1,3,7,15,31]
SEEDS=[0,1,2,3,4]
WEIGHTS=[.1,.2,.2,.2,.2,.1]
import importlib.metadata
_canonical=lambda x:json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()

def digest(value):return hashlib.sha256(_canonical(value)).hexdigest()

def versions():
 return {n:importlib.metadata.version(n) for n in ('numpy','torch','transformers','scikit-learn','tokenizers')}

def feature_matrix(records,embeddings,lookup):
 from sklearn.feature_extraction import FeatureHasher
 from sklearn.preprocessing import normalize
 from .experiments import FIELDS
 metadata=FeatureHasher(n_features=128,input_type='string',alternate_sign=False,dtype=np.float32).transform(
  [field+'='+str(row['subject'].get(field,'')) for field in FIELDS] for row in records)
 metadata=normalize(metadata,norm='l2').toarray()
 text=embeddings[[lookup[row['content']] for row in records]]
 return normalize(np.hstack([text,metadata]),norm='l2').astype(np.float32)

def expanded_training(records,indices):
 totals=defaultdict(int)
 for i in indices:totals[records[i]['pair_id']]+=records[i]['trials']
 expanded,labels,weights=[],[],[]
 for i in indices:
  row=records[i];s,n=row['successes'],row['trials']
  if not isinstance(s,int) or not isinstance(n,int) or not 0<=s<=n or n<=0:raise ValueError('invalid training counts')
  for label,count in ((1,s),(0,n-s)):
   if count:expanded.append(i);labels.append(label);weights.append(count/totals[row['pair_id']])
 weights=np.asarray(weights,dtype=np.float64);weights/=weights.mean()
 return np.asarray(expanded),np.asarray(labels),weights.astype(np.float32)

def fit_head(records,matrix,train_indices,target_indices,alpha):
 from sklearn.linear_model import SGDClassifier
 from scipy.special import expit
 if not train_indices:raise ValueError('empty training fold')
 indices,labels,weights=expanded_training(records,train_indices)
 if len(np.unique(labels))==1:
  p=(float(np.dot(weights,labels))+1)/(float(weights.sum())+2)
  coef=np.zeros(matrix.shape[1],dtype=np.float64);intercept=math.log(p/(1-p));epochs=0
 else:
  model=SGDClassifier(loss='log_loss',penalty='l2',alpha=alpha,max_iter=200,tol=1e-4,random_state=0)
  model.fit(matrix[indices],labels,sample_weight=weights)
  coef=model.coef_[0].astype(np.float64);intercept=float(model.intercept_[0]);epochs=int(model.n_iter_)
 prediction=expit(matrix[target_indices].astype(np.float64)@coef+intercept)
 return prediction,coef,intercept,{'epochs':epochs,'single_class_fallback':len(np.unique(labels))==1,
  'training_benchmarks':sorted({records[i]['benchmark'] for i in train_indices}),
  'train_records':len(train_indices),'expanded_rows':len(indices)}
