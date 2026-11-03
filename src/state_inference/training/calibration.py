"""Calibration fit only on a dedicated split, never on the test set."""
import math
import torch
from state_inference.metrics.binary import binary_metrics


def apply_platt(probabilities, parameters):
    output=[]
    for p in probabilities:
        p=min(1-1e-7,max(1e-7,p))
        z=parameters['scale']*math.log(p/(1-p))+parameters['bias']
        output.append(1/(1+math.exp(-max(-60,min(60,z)))))
    return output


def fit_calibration(labels, scores):
    if len(set(labels))!=2:raise ValueError('Calibration requires both classes')
    p=torch.tensor(scores,dtype=torch.float64).clamp(1e-7,1-1e-7)
    logits=torch.logit(p)
    y=torch.tensor(labels,dtype=torch.float64)
    raw=torch.tensor(math.log(math.expm1(1.0)),dtype=torch.float64,requires_grad=True)
    bias=torch.tensor(0.0,dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([raw,bias],lr=0.5,max_iter=80,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        scale=torch.nn.functional.softplus(raw)+1e-6
        loss=torch.nn.functional.binary_cross_entropy_with_logits(scale*logits+bias,y)+1e-4*(scale.square()+bias.square())
        loss.backward()
        return loss
    optimizer.step(closure)
    parameters={'scale':float((torch.nn.functional.softplus(raw)+1e-6).detach()),'bias':float(bias.detach()),
                'fit':'calibration split only; positive scale and 1e-4 L2'}
    calibrated=apply_platt(scores,parameters)
    candidates=sorted(set(calibrated))
    def f1(threshold):
        tp=sum(y==1 and p>=threshold for y,p in zip(labels,calibrated))
        fp=sum(y==0 and p>=threshold for y,p in zip(labels,calibrated))
        fn=sum(y==1 and p<threshold for y,p in zip(labels,calibrated))
        return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    parameters['threshold']=max(candidates,key=lambda t:(f1(t),t))
    return parameters


def metrics_at_threshold(labels,scores,threshold):
    result=binary_metrics(labels,scores)
    tp=sum(y==1 and p>=threshold for y,p in zip(labels,scores))
    fp=sum(y==0 and p>=threshold for y,p in zip(labels,scores))
    fn=sum(y==1 and p<threshold for y,p in zip(labels,scores))
    result['threshold']=threshold
    result['f1_at_calibrated_threshold']=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    return result
