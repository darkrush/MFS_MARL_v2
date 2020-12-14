import numpy as np
import torch
from torch.autograd import Variable


def onehot_from_logits(logits):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    if type(logits) == np.ndarray:
        logits = torch.FloatTensor(logits)

    dim = len(logits.shape) - 1
    argmax_acs = (logits == logits.max(dim, keepdim=True)[0]).float()
    return argmax_acs



def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""    
    y = (logits.cpu() + sample_gumbel(logits.shape, tens_type=type(logits.data))).cuda()

    dim = len(logits.shape) - 1
    return torch.nn.functional.softmax(y / temperature, dim=dim)

def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:       
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y
