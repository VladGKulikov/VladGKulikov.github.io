"""Behavioral attention checks embedded into the self-contained learner notebooks."""

def check_attention_contract(attention):
    """Compare outputs/gradients, batch broadcasting, and cached-query geometry.

    The callable returns an object with .output and .weights. No repository imports
    are needed here, so the notebook builder can embed this complete function.
    """
    import torch
    import torch.nn.functional as F
    torch.manual_seed(303)
    q=torch.randn(2,2,3,4,dtype=torch.float64,requires_grad=True)
    k=torch.randn(2,2,5,4,dtype=torch.float64,requires_grad=True)
    v=torch.randn(2,2,5,6,dtype=torch.float64,requires_grad=True)
    allowed=torch.arange(5)[None,:] <= (2+torch.arange(3))[:,None]
    explicit=attention(q,k,v,allowed).output
    library=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,dropout_p=0.0)
    torch.testing.assert_close(explicit,library,atol=1e-12,rtol=1e-12)
    probe=torch.randn_like(explicit)
    explicit_grad=torch.autograd.grad((explicit*probe).sum(),(q,k,v))
    library_grad=torch.autograd.grad((library*probe).sum(),(q,k,v))
    for a,b in zip(explicit_grad,library_grad):
        torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-11)

    # A mask can vary across a batch axis introduced only by V.
    bq=torch.zeros(1,2,1,dtype=torch.float64)
    bk=torch.zeros(1,3,1,dtype=torch.float64)
    bv=torch.tensor([[[1.],[2.],[3.]],[[4.],[5.],[6.]]],dtype=torch.float64)
    bm=torch.tensor([[[1,0,0],[0,1,0]],[[0,0,1],[1,1,0]]],dtype=torch.bool)
    batched=attention(bq,bk,bv,bm)
    torch.testing.assert_close(batched.output,torch.tensor([[[1.],[2.]],[[6.],[4.5]]],dtype=torch.float64))
    assert batched.weights.shape==(2,2,3)

    # Explicit zero rows also have finite gradients, rather than hidden NaNs.
    zq=q.detach().clone().requires_grad_()
    zk=k.detach().clone().requires_grad_()
    zv=v.detach().clone().requires_grad_()
    zm=allowed.clone();zm[1]=False
    zero=attention(zq,zk,zv,zm,empty_row='zero')
    assert torch.all(zero.output[...,1,:]==0)
    zg=torch.autograd.grad(zero.output.square().sum(),(zq,zk,zv))
    assert all(torch.isfinite(g).all() for g in zg)
    assert torch.all(zg[0][...,1,:]==0)

    # Last two queries have absolute positions 3 and 4, not local positions 0 and 1.
    fq=torch.randn(5,4,dtype=torch.float64)
    fk=torch.randn(5,4,dtype=torch.float64)
    fv=torch.randn(5,2,dtype=torch.float64)
    cm=torch.ones(5,5,dtype=torch.bool).tril()
    full=attention(fq,fk,fv,cm).output
    incremental=attention(fq[3:],fk,fv,cm[3:]).output
    torch.testing.assert_close(incremental,full[3:],atol=1e-12,rtol=1e-12)
    wrong=attention(fq[3:],fk,fv,torch.ones(2,5,dtype=torch.bool).tril()).output
    assert float((wrong-full[3:]).abs().max()) > 1e-3

    # Permuting unmasked input permutes output; it does not leave output rows fixed.
    perm=torch.tensor([2,0,4,1,3])
    original=attention(fq,fk,fv).output
    permuted=attention(fq[perm],fk[perm],fv[perm]).output
    torch.testing.assert_close(permuted,original[perm],atol=1e-12,rtol=1e-12)
    return {'output_max_error':float((explicit-library).abs().max().detach()),
            'gradient_max_error':max(float((a-b).abs().max()) for a,b in zip(explicit_grad,library_grad)),
            'wrong_cache_mask_max_error':float((wrong-full[3:]).abs().max()),
            'batch_broadcast':'PASS','zero_row_gradients':'PASS','permutation':'PASS'}
