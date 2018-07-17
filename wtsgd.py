import torch

def init_grad(model):
    '''
    Init to zero a list of tensors with the same shape of model.parameters()
    '''
    grad = []
    for param in model.parameters():      
        grad.append(torch.zeros_like(param))
    return grad   

def acc_grad(grad, model):
    '''
    Accumulate grad (computed on a minibatch)
    '''
    for g, param in zip(grad, model.parameters()):
        g += param.grad
    return grad

        
def acc_grad2(grad2, model):
    '''
    Accumulate squared grad (computed on a minibatch)
    '''
    for g, param in zip(grad2, model.parameters()):
        g += param.grad*param.grad
    return grad2

def compute_snr(grad, grad2, n):
    '''
    Compute snr
    '''  
    epsilon = 1e-8
    
    
    snr   = []
    for g, g2 in zip(grad, grad2):
       
        g_copy  = g.clone()
        g2_copy = g2.clone()
    
        # add a small quantity to squared grad (if zero) to avoid division by zero in err computation
        g2_copy[g2_copy==0] = epsilon
        
        # average over number of mini-batches in a large batch
        
        g_copy = g_copy/n    # this is what we called A  bar
        g2_copy = g2_copy/n  # this is what we called A2 bar
        
        # compute error        
        err = torch.sqrt( ( g2_copy - g_copy*g_copy )/ n )
        
        # compute signal to error ratio    
        snr.append( torch.abs(g_copy)/err ) 
        
    return snr