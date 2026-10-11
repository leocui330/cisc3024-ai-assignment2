import io
from pathlib import Path
import sys
import unittest
import torch
from torch.distributions import Normal, kl_divergence
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from heads import MAPHead, VBLLHead


class HeadTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(9)
        self.h = torch.randn(12,3,dtype=torch.float64)
        self.y = torch.arange(12)%2
        self.head = VBLLHead(3,2,.7).double()
        with torch.no_grad():
            self.head.mu.copy_(torch.randn_like(self.head.mu)*.3)
            self.head.rho.fill_(-1)

    def test_kl_matches_distribution_and_gradients(self):
        q = Normal(self.head.mu,self.head.variance.sqrt())
        p = Normal(torch.zeros_like(self.head.mu),torch.full_like(self.head.mu,.7))
        torch.testing.assert_close(self.head.kl_divergence(),kl_divergence(q,p).sum())
        self.head.loss(self.h,self.y,100).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.head.parameters()))

    def test_jensen_bound_and_bias_moments(self):
        m,v = self.head.moments(self.h)
        generator = torch.Generator().manual_seed(123)
        z = m[None]+v.sqrt()[None]*torch.randn(100000,*m.shape,generator=generator,dtype=m.dtype)
        mc_nll = (-z.log_softmax(-1)[:,:,0]* (self.y==0)[None]
                  -z.log_softmax(-1)[:,:,1]*(self.y==1)[None]).mean()
        bound = (torch.logsumexp(m+.5*v,-1)-m.gather(1,self.y[:,None]).squeeze()).mean()
        self.assertGreater(float((bound-mc_nll).detach()),-.003)
        h0 = torch.zeros(1,3,dtype=torch.float64)
        m0,v0=self.head.moments(h0)
        torch.testing.assert_close(m0[0],self.head.mu[:,-1])
        torch.testing.assert_close(v0[0],self.head.variance[:,-1])

    def test_probability_randomness_without_global_rng_change(self):
        rng=torch.get_rng_state()
        a=self.head.predict_proba(self.h,128,7)
        torch.testing.assert_close(rng,torch.get_rng_state())
        torch.testing.assert_close(a.sum(1),torch.ones(12,dtype=torch.float64))
        self.assertTrue((a>0).all())
        self.assertTrue(torch.equal(a,self.head.predict_proba(self.h,128,7)))
        self.assertFalse(torch.equal(a,self.head.predict_proba(self.h,128,8)))
        with self.assertRaises(ValueError):
            self.head.predict_proba(self.h,127)

    def test_zero_variance_limit(self):
        with torch.no_grad():
            self.head.rho.fill_(-50)
        torch.testing.assert_close(self.head.predict_proba(self.h),self.head.mean_proba(self.h),atol=1e-10,rtol=1e-10)

    def test_map_regularizer_and_checkpoint_reload(self):
        head=MAPHead(3,2,.7).double()
        with torch.no_grad():
            head.mu.copy_(self.head.mu)
        ce=torch.nn.functional.cross_entropy(head.moments(self.h)[0],self.y)
        torch.testing.assert_close(head.loss(self.h,self.y,100)-ce,head.mu.square().sum()/(2*.7**2*100))
        stream=io.BytesIO()
        torch.save(self.head.state_dict(),stream)
        stream.seek(0)
        other=VBLLHead(3,2,.7).double()
        other.load_state_dict(torch.load(stream,weights_only=True))
        self.assertTrue(torch.equal(other.predict_proba(self.h),self.head.predict_proba(self.h)))


if __name__=="__main__":
    unittest.main()
