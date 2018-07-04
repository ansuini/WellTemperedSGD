import torch

def init_grad(model):
    '''
    Init a list of variables with the same shape of model.parameters()
    '''
    grad = []
    for param in model.parameters():      
        grad.append(torch.zeros_like(param))
    return grad   

def acc_grad(grad, model):
    '''
    '''
    for g, param in zip(grad, model.parameters()):
        g += param.grad
        
    return grad

        
def acc_grad2(grad2, model):
    '''
    Elementwise multiplication
    '''
    for g, param in zip(grad2, model.parameters()):
        g += param.grad*param.grad
    return grad2

def compute_snr(grad, grad2, num_mb):
    '''
    Compute snr
    '''  
    snr   = []
    for g, g2 in zip(grad, grad2):
        
        # compute average of gradient on minibatches
        g = g/num_mb        
        
        # compute average of squared gradients
        g2 = g2/num_mb
        
        # compute error
        
        err = torch.sqrt( ( g2 - g*g )/ num_mb )
        
        # normalize error if zero
        
        # compute signal to error ratio
    
        snr.append( torch.abs(g)/err ) 
    return snr