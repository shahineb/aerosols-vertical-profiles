import torch
from gpytorch import kernels, lazy


class DeltaKernel(kernels.Kernel):
    """Compute covariance matrix based on delta kernel between inputs k(x1, x2) = δ(x1, x2)
    """
    def forward(self, x1, x2, **params):
        if torch.equal(x1, x2):
            output = lazy.DiagLazyTensor(torch.ones(x1.size(0), device=x1.device))
        else:
            dist = self.covar_dist(x1, x2) < torch.finfo(torch.float16).eps
            output = dist.float()
        return output
