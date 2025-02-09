import os
import json
import random
import math
import torch
import torch.utils.data
from convnext_v2_lora import *

class Counter:
    """
    A counter to track number of updates.
    """
    def __init__(self):
        self.count = 0
    def update(self):
        self.count += 1

class CMA(object):
    """
    A continual moving average for tracking loss updates.
    """
    def __init__(self):
        self.N = 0
        self.avg = 0.0
    def update(self, X):
        self.avg = (X + self.N * self.avg) / (self.N + 1)
        self.N = self.N + 1


def accuracy(output, target, topk=(1,), output_has_class_ids=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if not output_has_class_ids:
        output = torch.Tensor(output)
    else:
        output = torch.LongTensor(output)
    target = torch.LongTensor(target)
    with torch.no_grad():
        maxk = max(topk)
        batch_size = output.shape[0]
        if not output_has_class_ids:
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
        else:
            pred = output[:, :maxk].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def safe_load_dict(model, new_model_state, should_resume_all_params=False):
    old_model_state = model.state_dict()

    #for param_tensor in old_model_state:
    #    print(param_tensor, "\t", old_model_state[param_tensor].size())
    #print("\n")

    #for param_tensor in new_model_state:
    #    print(param_tensor, "\t", new_model_state[param_tensor].size())

    c = 0
    if should_resume_all_params:
        for old_name, old_param in old_model_state.items():
            assert old_name in list(new_model_state.keys()), "{} parameter is not present in resumed checkpoint".format(
                old_name)
    for name, param in new_model_state.items():
        n = name.split('.')
        beg = n[0]
        end = n[1:]
        if not beg == 'model':
            name = 'model.' + name ### Add this since name has to start with 'model'
        if beg == 'module':
            name = '.'.join(end)
        if name not in old_model_state:
            # print('%s not found in old model.' % name)
            continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        c += 1
        if old_model_state[name].shape != param.shape:
            print('Shape mismatch...ignoring %s' % name)
            if 'head' in name:
                #print(name)
                old_model_state[name][:1000].copy_(param)
            continue
        else:
            old_model_state[name].copy_(param)
    if c == 0:
        raise AssertionError('No previous ckpt names matched and the ckpt was not loaded properly.')


def build_classifier(classifier, classifier_ckpt, num_classes):

    if classifier_ckpt is None:
        print("Will not resume any checkpoints!")
    else:
        resumed = torch.load(classifier_ckpt)
        if 'state_dict' in resumed:
            state_dict_key = 'state_dict'
        else:
            state_dict_key = 'model' #'model_state'
        print("Resuming with {}".format(classifier_ckpt))
        classifier = eval(classifier)(pretrained_model=resumed[state_dict_key], num_classes=num_classes)
        #safe_load_dict(classifier, resumed[state_dict_key], should_resume_all_params=False)
    return classifier


def build_classifier_core(classifier, classifier_ckpt, num_classes): # for swav
    classifier = eval(classifier)(num_classes=num_classes)

    if classifier_ckpt is None:
        print("Will not resume any checkpoints!")
    else:
        resumed = torch.load(classifier_ckpt)
        if 'state_dict' in resumed:
            state_dict_key = 'state_dict'
        else:
            state_dict_key = 'model' #'model_state'
        print("Resuming with {}".format(classifier_ckpt))
        safe_load_dict(classifier, resumed[state_dict_key], should_resume_all_params=False)
    return classifier
