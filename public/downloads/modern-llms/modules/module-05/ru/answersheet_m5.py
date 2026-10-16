#!/usr/bin/env python3
"""Recompute and validate the frozen numerical contract for Module 5."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

MODULE_ROOT=Path(__file__).resolve().parents[1]
REFERENCE_PATH=MODULE_ROOT/'data/reference_metrics.json'
RU_LEGACY_PATH=MODULE_ROOT/'ru/viz/m5_answers.json'
EN_LEGACY_PATH=MODULE_ROOT/'en/viz/m5_answers.json'
KiB,MiB,GiB=2**10,2**20,2**30

def kv_bytes_per_token(layers:int,kv_heads:int,head_dim:int,bytes_per_value:int=2)->int:
    vals=(layers,kv_heads,head_dim,bytes_per_value)
    if any(isinstance(x,bool) or not isinstance(x,int) or x<=0 for x in vals):
        raise ValueError('all KV-cache dimensions must be positive integers')
    return 2*layers*kv_heads*head_dim*bytes_per_value

def compute_reference()->dict:
    context=131_072
    per_layer=2*8*64*2
    rows={
      'gpt_oss_gqa_all_full_equivalent':kv_bytes_per_token(36,8,64),
      'gpt_oss_mha_hypothetical':kv_bytes_per_token(36,64,64),
      'gpt_oss_mqa_hypothetical':kv_bytes_per_token(36,1,64),
    }
    logits=np.array([2.,1.,3.,.5]); values=np.array([1.,2.,3.,4.])
    l1=1+math.exp(-1); n1=1+2*math.exp(-1); alpha=math.exp(-1)
    l2=alpha*l1+1+math.exp(-2.5); n2=alpha*n1+3+4*math.exp(-2.5)
    w=np.exp(logits-logits.max()); w/=w.sum(); out=n2/l2
    assert abs(out-w@values)<1e-12
    mla=(512+64)*61*2; mha=2*61*128*128*2
    full=18*context*per_layer; sliding=18*128*per_layer; all_full=rows['gpt_oss_gqa_all_full_equivalent']*context
    return {
      'schema_version':2,'context_tokens':context,'kv_rows':rows,
      'gpt_oss_hybrid':{
        'layers':36,'full_layers':18,'sliding_layers':18,'window':128,
        'per_layer_bytes_per_token':per_layer,'full_bytes':full,'sliding_bytes':sliding,
        'hybrid_total_bytes':full+sliding,'all_full_bytes':all_full,
      },
      'online_softmax_trace':{'l1':l1,'n1':n1,'alpha':alpha,'l2':l2,'n2':n2,'output':out},
      'mla':{'bytes_per_token':mla,'mha_equivalent_bytes_per_token':mha,'raw_element_ratio':mha/mla},
      'score_matrix':{
        'n_4096_fp32_bytes':4096*4096*4,'n_8192_fp32_bytes':8192*8192*4,
        'tile_128_fp32_bytes':128*128*4,
      },
    }

def assert_nested_close(actual,expected,path='root'):
    if isinstance(expected,dict):
        if set(actual)!=set(expected): raise AssertionError(f'{path}: keys differ: {set(actual)^set(expected)}')
        for k in expected: assert_nested_close(actual[k],expected[k],f'{path}.{k}')
    elif isinstance(expected,float):
        if not math.isclose(float(actual),expected,rel_tol=0,abs_tol=1e-12): raise AssertionError(f'{path}: {actual!r} != {expected!r}')
    elif actual!=expected: raise AssertionError(f'{path}: {actual!r} != {expected!r}')

def legacy_payload(r,lang):
    labels={
      'gpt_oss_gqa_all_full_equivalent':'gpt-oss GQA all-full equivalent',
      'gpt_oss_mha_hypothetical':'gpt-oss hypothetical MHA',
      'gpt_oss_mqa_hypothetical':'gpt-oss hypothetical MQA',
    }
    t=r['online_softmax_trace']; h=r['gpt_oss_hybrid']
    return {'kv_rows':{labels[k]:v for k,v in r['kv_rows'].items()},
      'trace':{'l1':t['l1'],'n1':t['n1'],'a':t['alpha'],'l2':t['l2'],'n2':t['n2'],'O':t['output']},
      'mla':{'bpt':r['mla']['bytes_per_token'],'mha_eq':r['mla']['mha_equivalent_bytes_per_token'],'raw_ratio':r['mla']['raw_element_ratio']},
      'layers':{'full':h['full_bytes'],'slide':h['sliding_bytes'],'hybrid':h['hybrid_total_bytes'],'all_full':h['all_full_bytes']}}

def print_report(r):
    context=r['context_tokens']; print('== A. KV bytes/token, bf16 ==')
    for k,v in r['kv_rows'].items(): print(f'{k:39s}: {v:>8,} B = {v/KiB:6.1f} KiB/token; all-full @131072 -> {v*context/GiB:7.3f} GiB')
    h=r['gpt_oss_hybrid']; print('\n== B. gpt-oss hybrid cache ==')
    print(f"18 full={h['full_bytes']/GiB:.3f} GiB; 18 sliding(128)={h['sliding_bytes']/MiB:.3f} MiB; hybrid={h['hybrid_total_bytes']/GiB:.6f} GiB; all-full={h['all_full_bytes']/GiB:.3f} GiB")
    m=r['mla']; print('\n== C. MLA raw storage geometry ==')
    print(f"MLA={m['bytes_per_token']/KiB:.3f} KiB/token; MHA baseline={m['mha_equivalent_bytes_per_token']/MiB:.3f} MiB/token; raw ratio={m['raw_element_ratio']:.6f}x")
    print('\n== D. Online softmax =='); print(json.dumps(r['online_softmax_trace'],indent=2))
    s=r['score_matrix']; print('\n== E. Score working sets ==')
    print(f"4096 fp32={s['n_4096_fp32_bytes']/MiB:.1f} MiB/head; 8192 fp32={s['n_8192_fp32_bytes']/MiB:.1f} MiB/head; 128x128 tile={s['tile_128_fp32_bytes']/KiB:.1f} KiB")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--write',action='store_true'); args=ap.parse_args()
    c=compute_reference()
    if args.write:
        REFERENCE_PATH.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        RU_LEGACY_PATH.write_text(json.dumps(legacy_payload(c,'ru'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        EN_LEGACY_PATH.write_text(json.dumps(legacy_payload(c,'en'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    stored=json.loads(REFERENCE_PATH.read_text(encoding='utf-8')); assert_nested_close(c,stored)
    print_report(c); print('\nModule-5 numerical contract: PASS')
if __name__=='__main__': main()
