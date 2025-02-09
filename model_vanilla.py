import os
import time
import shutil
import sys
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pickle
import copy
from copy import deepcopy
import utils as utils
from retrieve_any_layer import ModelWrapper
sys.setrecursionlimit(10000)

def _set_seed(seed):
    print("Set seed", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class VModel(object):
    def __init__(self, num_classes, classifier_G='ConvNeXtNet', extract_features_from='model.downsample_layers.2',
            classifier_F='ConvNeXt_block2', hidden_dim=384, classifier_ckpt=None, weight_decay=5e-2,
            base_init_classes=1000, class_increment=73, max_buffer_size=None, lr=0.001, seed=1993):

        # make the classifier
        self.classifier_F = utils.build_classifier(classifier_F, classifier_ckpt, num_classes=num_classes) # plastic part of DNN
        core_model = utils.build_classifier_core(classifier_G, classifier_ckpt, num_classes=num_classes)
        self.classifier_G = ModelWrapper(core_model, output_layer_names=[extract_features_from], return_single=True) # frozen part (feature extractor)
        self.hidden_dim = hidden_dim
        self.classifier_ckpt = classifier_ckpt
        self.num_classes = num_classes  # 1365
        self.base_init_classes = base_init_classes
        self.class_increment = class_increment
        self.max_buffer_size = max_buffer_size
        self.lr = lr
        self.weight_decay = weight_decay
        ## Set seed
        _set_seed(seed)


    #### ///// LayerWiseLR ///// ####
    def get_layerwise_params(self, classifier, lr, decay=0.99):
        trainable_params = []
        layer_names = []
        lr_mult = decay
        for idx, (name, param) in enumerate(classifier.named_parameters()):
            layer_names.append(name)
        # reverse layers
        layer_names.reverse()
        # store params & learning rates
        for idx, name in enumerate(layer_names):
            # append layer parameters
            trainable_params += [{'params': [p for n, p in classifier.named_parameters() if n == name and p.requires_grad],
                            'lr': lr}]
            # update learning rate
            lr *= lr_mult
        return trainable_params

    ##### ------------------------- #####
    ##### ----- UPDATE BUFFER ----- #####
    ##### ------------------------- #####
    def update_buffer(self, curr_loader, latent_dict, rehearsal_ixs, class_id_to_item_ix_dict, counter):
        start_time = time.time()
        for batch_images, batch_labels, batch_item_ixs in curr_loader: # New classes
            # put codes and labels into buffer (dictionary)
            for x, y, item_ix in zip(batch_images, batch_labels, batch_item_ixs): # x dim: 1x7x7x32
                # Add new data index and label to dict (new class)
                latent_dict[int(item_ix.numpy())] = [y.numpy()]
                rehearsal_ixs.append(int(item_ix.numpy()))
                class_id_to_item_ix_dict[int(y.numpy())].append(int(item_ix.numpy()))
                # if buffer is full, randomly replace previous example from class with most samples
                if self.max_buffer_size is not None and counter.count >= self.max_buffer_size:
                    # class with most samples and random item_ix from it
                    max_key = max(class_id_to_item_ix_dict, key=lambda x: len(class_id_to_item_ix_dict[x]))
                    max_class_list = class_id_to_item_ix_dict[max_key]
                    rand_item_ix = random.choice(max_class_list)
                    # remove the random_item_ix from all buffer references
                    max_class_list.remove(rand_item_ix)
                    latent_dict.pop(rand_item_ix)
                    rehearsal_ixs.remove(rand_item_ix)
                else:
                    counter.update()

        spent_time = int((time.time() - start_time) / 60)  # in minutes
        print("\nTime spent in buffer update process (in mins):", spent_time)
        return latent_dict, rehearsal_ixs, class_id_to_item_ix_dict


    # ---------------------------------------------------------- #
    # ------ Train on Places365 (CL) + ImageNet1K (Base) ------- #
    # ---------------------------------------------------------- #
    ## First session

    def train_s1(self, train_loader_new, train_loader_old, test_loader_new, test_loader_old,
        num_iters, val_acc_all, val_acc_old, val_acc_new, val_acc_base, wandb):

        start_time = time.time()
        total_loss = utils.CMA()
        params = self.get_layerwise_params(self.classifier_F, self.lr, 0.9)
        self.classifier_G.eval()
        self.classifier_G.cuda()
        classifier_F = self.classifier_F.cuda()
        classifier_F.train()
        criterion = nn.CrossEntropyLoss().cuda()
        vf = 100 #10 #100 # validation frequency for metrics
        num_iter = num_iters
        ## Optimizer Initialization
        optimizer = optim.AdamW(params, weight_decay=self.weight_decay)


        for i, (data1, data2) in enumerate(zip(train_loader_new, train_loader_old)):
            batch_x1, batch_y1 = data1[0], data1[1] # new
            batch_x2, batch_y2 = data2[0], data2[1] # old
            batch_y1 = batch_y1 + self.base_init_classes # batch_y1 + 1000 # new labels
            x = torch.cat((batch_x1, batch_x2), axis=0)
            y = torch.cat((batch_y1, batch_y2), axis=0)
            data = self.classifier_G(x.cuda())
            output = classifier_F(data)  # data dim: N x 384 x 14 x 14, output dim: N x 1365
            loss = criterion(output, y.cuda())

            ## Optimizer & BackProp
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.update(loss.item())

            ## Log
            if (i+1) % vf == 0:
                ## Compute validation accuracy
                ## old
                probas_old, true_old = self.predict(test_loader_old)
                top1_old, _ = utils.accuracy(probas_old, true_old, topk=(1, 5))
                ## new
                probas_new, true_new = self.predict(test_loader_new)
                true_new = true_new + self.base_init_classes # true_new + 1000 # new lables
                top1_new, _ = utils.accuracy(probas_new, true_new, topk=(1, 5))
                ## All
                probas_all = np.concatenate((probas_old, probas_new), axis=0)
                true_all = np.concatenate((true_old, true_new), axis=0)
                top1_all, _ = utils.accuracy(probas_all, true_all, topk=(1, 5))
                ## wandb
                wandb.log({
                "base_acc1": top1_old,
                "old_acc1": top1_old,
                "new_acc1": top1_new,
                "all_acc1": top1_all
                })
                print("All: %1.2f" % top1_all, "--Old: %1.2f" % top1_old, "--New: %1.2f" % top1_new)
                val_acc_all = np.append(val_acc_all, top1_all)
                val_acc_old = np.append(val_acc_old, top1_old)
                val_acc_new = np.append(val_acc_new, top1_new)
                val_acc_base = np.append(val_acc_base, top1_old)

            if (i+1) % 250 == 0 or i == 0 or (i+1) == num_iter:
                print("Iteration:", (i+1), "--Loss: %1.5f" % total_loss.avg)

        spent_time = int((time.time() - start_time) / 60)  # in minutes
        print("\nTime Spent in Updating DNN (in mins):", spent_time)

        return val_acc_all, val_acc_old, val_acc_new, val_acc_base


    ### After first session
    def train(self, train_loader_new, train_loader_old, train_loader_prev, test_loader_new, test_loader_old,
        test_loader_prev, num_iters, val_acc_all, val_acc_old, val_acc_new, val_acc_base, old_class_list, wandb):

        start_time = time.time()
        total_loss = utils.CMA()
        params = self.get_layerwise_params(self.classifier_F, self.lr, 0.9)
        self.classifier_G.eval()
        self.classifier_G.cuda()
        classifier_F = self.classifier_F.cuda()
        classifier_F.train()
        criterion = nn.CrossEntropyLoss().cuda()
        vf = 100 #10 #100 # validation frequency for metrics
        num_iter = num_iters
        ## Optimizer
        optimizer = optim.AdamW(params, weight_decay=self.weight_decay)


        for i, (data1, data2, data3) in enumerate(zip(train_loader_new, train_loader_old, train_loader_prev)):
            batch_x1, batch_y1, = data1[0], data1[1] # new
            batch_x2, batch_y2 = data2[0], data2[1] # old
            batch_x3, batch_y3 = data3[0], data3[1] # prev
            batch_y1 = batch_y1 + self.base_init_classes # batch_y1 + 1000 # new labels
            batch_y3 = batch_y3 + self.base_init_classes # batch_y3 + 1000 # new previous labels
            x = torch.cat((batch_x1, batch_x2, batch_x3), axis=0)
            y = torch.cat((batch_y1, batch_y2, batch_y3), axis=0)
            data = self.classifier_G(x.cuda())
            output = classifier_F(data)  # data dim: N x 384 x 14 x 14, output dim: N x 1365
            loss = criterion(output, y.cuda())

            ## Optimizer and Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.update(loss.item())

            ## Log
            if (i+1) % vf == 0:
                ## compute validation accuracy
                ## base
                probas_base, true_base = self.predict(test_loader_old)
                top1_base, _ = utils.accuracy(probas_base, true_base, topk=(1, 5))
                ## Prev
                probas_prev, true_prev = self.predict(test_loader_prev)
                true_prev = true_prev + self.base_init_classes # true_prev + 1000 # prev lables
                top1_prev, _ = utils.accuracy(probas_prev, true_prev, topk=(1, 5))
                ## Old = base + prev
                probas_old = np.concatenate((probas_base, probas_prev), axis=0)
                true_old = np.concatenate((true_base, true_prev), axis=0)
                top1_old, _ = utils.accuracy(probas_old, true_old, topk=(1, 5))
                ## new
                probas_new, true_new = self.predict(test_loader_new)
                true_new = true_new + self.base_init_classes # true_new + 1000 # new lables
                top1_new, _ = utils.accuracy(probas_new, true_new, topk=(1, 5))
                ## All
                probas_all = np.concatenate((probas_old, probas_new), axis=0)
                true_all = np.concatenate((true_old, true_new), axis=0)
                top1_all, _ = utils.accuracy(probas_all, true_all, topk=(1, 5))
                ## wandb
                wandb.log({
                "base_acc1": top1_base,
                "old_acc1": top1_old,
                "new_acc1": top1_new,
                "all_acc1": top1_all
                })
                print("All: %1.2f" % top1_all, "--Old: %1.2f" % top1_old, "--New: %1.2f" % top1_new)
                val_acc_all = np.append(val_acc_all, top1_all)
                val_acc_old = np.append(val_acc_old, top1_old)
                val_acc_new = np.append(val_acc_new, top1_new)
                val_acc_base = np.append(val_acc_base, top1_base)

            if (i+1) % 250 == 0 or i == 0 or (i+1) == num_iter:
                print("Iter:", (i+1), "--Loss: %1.5f" % total_loss.avg)

        spent_time = int((time.time() - start_time) / 60)  # in minutes
        print("\nTime Spent in Updating DNN (in mins):", spent_time)

        return val_acc_all, val_acc_old, val_acc_new, val_acc_base


    ### ------------------------- ###
    ### ---------- Test --------- ###
    ### ------------------------- ###
    def predict(self, data_loader):
        with torch.no_grad():
            self.classifier_F.eval().cuda()
            self.classifier_G.eval().cuda()
            probas = torch.zeros((len(data_loader.dataset), self.num_classes), dtype=torch.float64)
            all_lbls = torch.zeros((len(data_loader.dataset)))
            #print("\nNumber of samples in test set:", len(data_loader.dataset))
            start_ix = 0

            for batch_ix, batch in enumerate(data_loader):
                batch_x, batch_lbls = batch[0], batch[1]
                batch_x = batch_x.cuda()
                ## Get G features
                data_batch = self.classifier_G(batch_x)
                batch_lbls = batch_lbls.cuda()
                logits = self.classifier_F(data_batch)
                end_ix = start_ix + len(batch_x)
                probas[start_ix:end_ix] = F.softmax(logits.data, dim=1)
                all_lbls[start_ix:end_ix] = batch_lbls.squeeze()
                start_ix = end_ix

        return probas.numpy(), all_lbls.int().numpy()


    def resume(self, inc, resume_full_path):
        print(f'\nResuming DNN model from {resume_full_path}')
        state = torch.load(os.path.join('./' + resume_full_path, 'best_' + resume_full_path + '.pth'))
        utils.safe_load_dict(self.classifier_F, state['state_dict'], should_resume_all_params=False)
        ## sanity check whether two checkpoints are similar ##
        old_state=state['state_dict']
        new_state = self.classifier_F.state_dict()
        for k in old_state: # pretrained checkpoint
            assert torch.equal(old_state[k].cpu(), new_state[k[len("module."):]]), k
        print("\nSuccessfully performed sanity check!!")

        # Load Parameters
        with open(os.path.join(resume_full_path, 'buffer_%d.pkl' % inc), 'rb') as f:
            d = pickle.load(f)

        return d['latent_dict'], d['rehearsal_ixs'], d['class_id_to_item_ix_dict']


        ### << THE END >> ###
