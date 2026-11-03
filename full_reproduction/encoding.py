"""Offline pinned E5 prefix and head-tail rebuild; no workspace imports."""
import time
import numpy as np
import torch
from safety import MANIFEST, dump, resources
from science.head_tail_20261016.core import tail_body_ids

def rebuild(texts,encoder,output,old_head,old_affected,old_tail):
    import transformers.utils.import_utils as optional
    optional._torchvision_available=False
    from transformers import AutoModel, AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(encoder),local_files_only=True,trust_remote_code=False,token=False)
    model=AutoModel.from_pretrained(str(encoder),local_files_only=True,trust_remote_code=False,use_safetensors=True,token=False).to('cpu',dtype=torch.float32).eval()
    model.requires_grad_(False);torch.set_num_threads(2)
    directory=output/'fresh_encoding';directory.mkdir(exist_ok=True);started=time.monotonic()
    def pool(tokens):
        with torch.inference_mode():
            last=model(**tokens).last_hidden_state;mask=tokens['attention_mask'].unsqueeze(-1).to(last.dtype)
            value=(last*mask).sum(1)/mask.sum(1).clamp(min=1)
            return torch.nn.functional.normalize(value,p=2,dim=1).cpu().numpy().astype(np.float32)
    heads=np.empty((len(texts),384),np.float32)
    for begin in range(0,len(texts),8):
        tokens=tokenizer(['query: '+text for text in texts[begin:begin+8]],padding=True,truncation=True,max_length=512,return_tensors='pt')
        heads[begin:begin+8]=pool(tokens)
        if begin%128==0:
            dump(directory/'progress.json',dict(stage='prefix',completed=min(begin+8,len(texts)),total=len(texts),seconds=time.monotonic()-started,resources=resources(output)))
    prefix=tuple(tokenizer('query:',add_special_tokens=False)['input_ids'])
    if not prefix or tokenizer.num_special_tokens_to_add(pair=False)!=2:raise ValueError('frozen tokenizer prefix/special-token contract differs')
    affected=[];windows=[]
    for index,text in enumerate(texts):
        body=tokenizer('query: '+text,add_special_tokens=False,truncation=False)['input_ids']
        if tuple(body[:len(prefix)])!=prefix:raise ValueError('prefix binding differs')
        if len(tokenizer.build_inputs_with_special_tokens(body))<=512:continue
        values=tokenizer.build_inputs_with_special_tokens(list(tail_body_ids(body,len(prefix),510)))
        if len(values)>512:raise ValueError('tail exceeds512 tokens')
        affected.append(index);windows.append(values)
    affected=np.asarray(affected,np.int64);tails=np.empty((len(affected),384),np.float32)
    for begin in range(0,len(affected),8):
        chunk=windows[begin:begin+8];width=max(map(len,chunk));ids=np.full((len(chunk),width),tokenizer.pad_token_id,np.int64);attention=np.zeros_like(ids)
        for row_index,values in enumerate(chunk):ids[row_index,:len(values)]=values;attention[row_index,:len(values)]=1
        tails[begin:begin+8]=pool(dict(input_ids=torch.from_numpy(ids),attention_mask=torch.from_numpy(attention)))
        if begin%128==0:dump(directory/'progress.json',dict(stage='tail',completed=min(begin+8,len(affected)),total=len(affected),resources=resources(output)))
    if not np.array_equal(affected,old_affected):raise ValueError('fresh tokenizer tail membership differs from frozen source')
    for values in (heads,tails):
        if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values,axis=1),1,atol=2e-4):raise ValueError('invalid rebuilt normalized embeddings')
    np.savez_compressed(directory/'prefix_vectors.npz',vectors=heads)
    np.savez_compressed(directory/'tail_vectors.npz',affected_indices=affected,vectors=tails,completed=np.ones(len(affected),np.bool_))
    dump(directory/'receipt.json',dict(model_id=MANIFEST['encoder']['model_id'],revision=MANIFEST['encoder']['revision'],prefix='query: ',pooling='masked_mean_then_L2',max_tokens=512,cpu=True,batch_size=8,head_max_abs_error=float(np.max(np.abs(heads-old_head))),tail_max_abs_error=float(np.max(np.abs(tails-old_tail))),seconds=time.monotonic()-started,new_model_training=False,platform_parity='numeric differences reported; no byte-identity claim'))
    del model
    return heads,affected,tails
