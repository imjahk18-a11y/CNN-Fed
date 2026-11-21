import torch
import torch.nn.functional as F

from models.voxel_3dcnn import nonoverlapping_max_pool
from training.train import masked_cross_entropy


def test_equivalent_pool_and_loss_gradients():
    g = torch.Generator().manual_seed(32)
    x = torch.randint(0,3,(2,3,7,8,9),generator=g).float().requires_grad_()
    y = x.detach().clone().requires_grad_()
    ref = F.max_pool3d(x,2)
    output = nonoverlapping_max_pool(y)
    torch.testing.assert_close(ref, output)
    grad = torch.randn(ref.shape,generator=g)
    ref.backward(grad)
    output.backward(grad)
    torch.testing.assert_close(x.grad,y.grad)
    labels = torch.randint(0,3,(2,7,8,9),generator=g)
    labels[:,0] = -100
    a,b = [x.detach().clone().requires_grad_() for _ in range(2)]
    reference = F.cross_entropy(a,labels)
    masked = masked_cross_entropy(b,labels)
    reference.backward()
    masked.backward()
    torch.testing.assert_close(reference,masked)
    torch.testing.assert_close(a.grad,b.grad)
