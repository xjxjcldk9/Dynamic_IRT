from gibbs_sampling import NormalSampler

def test_normal_sampler():
    a = NormalSampler(1, 1)
    b = NormalSampler(2, 1)
    c = NormalSampler(3, 1)
    
    d = a * b * c 
    assert d.mu == 2
    assert d.sigma2 == 1/3