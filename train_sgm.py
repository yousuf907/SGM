import os
import argparse
import time
import torch
import json
import numpy as np
import wandb
from model_sgm import SModel
import utils_data as utils_data
import utils as utils
from collections import defaultdict
torch.multiprocessing.set_sharing_strategy('file_system')

#### ////////// Proposed SGM Model (Weight Init + Soft Targets + LoRA + OOCF) ////////// ####

## Data Loader Function
def get_data_loader(images_dir, label_dir, split, seed, min_class, max_class, batch_size=128, num_iter=0,
                    return_item_ix=False, curr_idx=None, shuffle=False):

    data_loader = utils_data.get_places_data_loader(images_dir + '/' + split, label_dir, split, batch_size=batch_size,
        num_iter=num_iter, shuffle=shuffle, min_class=min_class, max_class=max_class,
        return_item_ix=return_item_ix, seed=seed, curr_idx=curr_idx)

    return data_loader

def get_data_loader_old(images_dir, label_dir, split, min_class, max_class, batch_size=128,
                        return_item_ix=False, curr_idx=None):

    data_loader = utils_data.get_imagenet_data_loader(images_dir + '/' + split, label_dir, split, batch_size=batch_size,
        shuffle=False, min_class=min_class, max_class=max_class, return_item_ix=return_item_ix, curr_idx=curr_idx)

    return data_loader


def continual_learning(args, sgm, wandb):

    counter = utils.Counter()
    start_time=time.time()
    num_session =0
    latent_dict = {}
    rehearsal_ixs = []
    class_id_to_item_ix_dict = defaultdict(list)
    num_iter = args.num_iter # 1200 ## Number of training iteration during each CL session / CL task / batch update
    comp_budget = args.batch_size * args.num_iter # compute budget for a rehearsal/ training session
    print("Number of training iterations per session:", num_iter)

    if args.resume_full_path is not None:
        # load in previous model to continue training
        latent_dict, rehearsal_ixs, class_id_to_item_ix_dict = sgm.resume(args.base_init_classes, args.resume_full_path)
        ## validate performance from previous increment
        print('Previous model loaded...computing previous accuracy as sanity check...')
        init_test_loader = get_data_loader(args.images_dir, args.label_dir, 'val', args.min_class,
                    args.base_init_classes, batch_size=args.batch_size) #0-C
        print('\nComputing accuracies...')
        probas_base, true_base = sgm.predict(init_test_loader)
        top1_base, top5_base = utils.accuracy(probas_base, true_base, topk=(1, 5))
        print('\nAccuracy on past classes: top1=%0.2f%% -- top5=%0.2f%%' % (top1_base, top5_base))
        counter.count = len(rehearsal_ixs)
    else:
        print('\nPerforming base initialization...')
        ################ Load ImageNet1K Subset Indices (constrained buffer setting)
        all_base_idx=[]
        all_idxs = np.load('./imagenet_files/rehearsal_ixs_ImageNet_18600.npy') # Budget 24K (18600 ImageNet + 5400 Places)
        #all_idxs = np.load('./imagenet_files/rehearsal_ixs_ImageNet_153600.npy') # Budget 192K (153600 ImageNet + 38400 Places)
        #all_idxs = np.load('./imagenet_files/rehearsal_ixs_ImageNet_38400.npy') # Budget 48K (38400 ImageNet + 9600 Places)

        q = int(comp_budget / len(all_idxs)) + 1
        for i in range(q):
            all_base_idx = np.append(all_base_idx, all_idxs) # augment to create the squence

        all_base_idx = np.array(all_base_idx[:comp_budget], dtype=np.int32)
        assert len(all_base_idx) == comp_budget

        ## ImageNet-1K Evaluation ## BASE-INIT CLASSES
        print('Computing base accuracies...')
        test_loader_old = get_data_loader_old(args.images_dir_old, args.label_dir_old, 'val', args.min_class,
                            args.base_init_classes, batch_size=args.batch_size)
        print("Number of test samples in ImageNet loader:", len(test_loader_old.dataset))
        probas_base, true_base = sgm.predict(test_loader_old)
        top1_base, top5_base = utils.accuracy(probas_base, true_base, topk=(1, 5))
        print('Accuracy on base_init classes: top1=%0.2f%% -- top5=%0.2f%%' % (top1_base, top5_base))

    ## Continual Learning
    last_class = args.cl_max_class #365
    session_tot = int((last_class - args.cl_min_class) / args.class_increment)
    print('\nBeginning SGM Training...')
    print("Total number of sessions:", session_tot)
    val_acc_all=[]
    val_acc_old=[]
    val_acc_new=[]
    val_acc_base=[]
    new_class_list = []
    num_recent_stuff = 0

    for class_ix in range(args.cl_min_class, args.cl_max_class, args.class_increment):
        max_class = class_ix + args.class_increment
        num_session += 1
        print('\nCurrent Session ', num_session)
        print('Training classes {}-{}.'.format(class_ix, max_class))

        curr_loader = get_data_loader(args.images_dir, args.label_dir, 'train', args.seed, class_ix, max_class,
                batch_size=args.batch_size, num_iter=0, return_item_ix=True, shuffle=False) # 0-max_class
        print("\nNumber of training samples in Places365 loader for weight init:", len(curr_loader .dataset))

        if num_session > 1:
            train_loader_prev = get_data_loader(args.images_dir, args.label_dir, 'train', args.seed, args.cl_min_class, class_ix,
                batch_size=32, num_iter=num_iter, return_item_ix=False, curr_idx=replay_ixs, shuffle=True) # 0-class_ix
            print("\nNumber of training samples in Places365 previous loader:", len(train_loader_prev.dataset))

        latent_dict, rehearsal_ixs, class_id_to_item_ix_dict = sgm.update_buffer(curr_loader, latent_dict,
                      rehearsal_ixs, class_id_to_item_ix_dict, counter)
        replay_ixs = np.array(rehearsal_ixs, dtype=np.int32)

        test_loader_new = get_data_loader(args.images_dir, args.label_dir, 'val', args.seed, class_ix, max_class,
                                batch_size=args.batch_size)
        print("\nNumber of test samples in Places365 new loader:", len(test_loader_new.dataset))

        if num_session > 1:
            train_loader_new = get_data_loader(args.images_dir, args.label_dir, 'train', args.seed, class_ix, max_class,
                batch_size=96, num_iter=0, return_item_ix=False, curr_idx=None, shuffle=True) # 0-max_class
            print("\nNumber of training samples in Places365 new loader:", len(train_loader_new.dataset))

            test_loader_prev = get_data_loader(args.images_dir, args.label_dir, 'val', args.seed, args.cl_min_class, class_ix,
                                batch_size=args.batch_size)
            print("\nNumber of test samples in Places365 previous loader:", len(test_loader_prev.dataset))
        else:
            train_loader_new = get_data_loader(args.images_dir, args.label_dir, 'train', args.seed, class_ix, max_class,
                batch_size=args.batch_size, num_iter=0, return_item_ix=False, curr_idx=None, shuffle=True) # 0-max_class
            print("\nNumber of training samples in Places365 new loader:", len(train_loader_new.dataset))

        ## Old Classes : ImageNet-1K
        np.random.shuffle(all_base_idx) ## Randomize/Shuffle
        train_loader_old = get_data_loader_old(args.images_dir_old, args.label_dir_old, 'train',
             min_class=None, max_class=None, batch_size=args.batch_size, return_item_ix=False, curr_idx=all_base_idx)
        print("\nNumber of training samples in ImageNet loader:", len(train_loader_old.dataset))

        #///////////////////////////////////////////////////// TRAINING PHASE /////////////////////////////////////////////

        ### /// Data-driven Weight Initialization /// ###
        print("\nPerforming Data-driven Weight Intialization")
        sgm.init_new_weights(curr_loader)

        ### Pre Session Evaluation ###
        if num_session > 1:
            ## base (1000)
            probas_base, true_base = sgm.predict(test_loader_old)
            top1_base, top5_base = utils.accuracy(probas_base, true_base, topk=(1, 5))
            ## Prev (73)
            probas_prev, true_prev = sgm.predict(test_loader_prev)
            true_prev = true_prev + args.base_init_classes # prev lables
            top1_prev, top5_prev = utils.accuracy(probas_prev, true_prev, topk=(1, 5))
            ## Old = base + prev (1073)
            probas_old = np.concatenate((probas_base, probas_prev), axis=0)
            true_old = np.concatenate((true_base, true_prev), axis=0)
            top1_old, top5_old = utils.accuracy(probas_old, true_old, topk=(1, 5))
            ## new (73)
            probas_new, true_new = sgm.predict(test_loader_new)
            true_new = true_new + args.base_init_classes # new lables
            top1_new, top5_new = utils.accuracy(probas_new, true_new, topk=(1, 5))
            ## All (1146)
            probas_all = np.concatenate((probas_old, probas_new), axis=0)
            true_all = np.concatenate((true_old, true_new), axis=0)
            top1_all, _ = utils.accuracy(probas_all, true_all, topk=(1, 5))
            old_class_list = np.unique(true_old)
        else:
            ## OLD Classes
            probas_old, true_old = sgm.predict(test_loader_old)
            top1_old, top5_old = utils.accuracy(probas_old, true_old, topk=(1, 5))
            ## NEW Classes
            probas_new, true_new = sgm.predict(test_loader_new)
            true_new = true_new + args.base_init_classes # new lables
            top1_new, top5_new = utils.accuracy(probas_new, true_new, topk=(1, 5))
            ## All Classes
            probas_all = np.concatenate((probas_old, probas_new), axis=0)
            true_all = np.concatenate((true_old, true_new), axis=0)
            top1_all, top5_all = utils.accuracy(probas_all, true_all, topk=(1, 5))
            top1_base=top1_old
            top5_base=top5_old

        print('Pre-session accuracy [BASE CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_base, top5_base))
        print('Pre-session accuracy [OLD CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_old, top5_old))
        print('Pre-session accuracy [NEW CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_new, top5_new))
        print('Pre-session accuracy [ALL CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_all, top5_all))
        val_acc_all = np.append(val_acc_all, top1_all)
        val_acc_old = np.append(val_acc_old, top1_old)
        val_acc_new = np.append(val_acc_new, top1_new)
        val_acc_base = np.append(val_acc_base, top1_base)
        ####

        ## wandb
        wandb.log({
        "base_acc1": top1_base,
        "old_acc1": top1_old,
        "new_acc1": top1_new,
        "all_acc1": top1_all
        })

        print("Training is begining..")
        if num_session == 1:
            val_acc_all, val_acc_old, val_acc_new, val_acc_base = sgm.train_s1(train_loader_new, train_loader_old,
             test_loader_new, test_loader_old, num_iter, val_acc_all, val_acc_old, val_acc_new, val_acc_base, wandb)
        else:
            val_acc_all, val_acc_old, val_acc_new, val_acc_base = sgm.train(train_loader_new, train_loader_old,
             train_loader_prev, test_loader_new, test_loader_old, test_loader_prev, num_iter,
             val_acc_all, val_acc_old, val_acc_new, val_acc_base, old_class_list, wandb)


    ## Save results
    exp_dir = args.save_dir
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    np.save(os.path.join(exp_dir, 'val_all_sgm.npy'), val_acc_all)
    np.save(os.path.join(exp_dir, 'val_old_sgm.npy'), val_acc_old)
    np.save(os.path.join(exp_dir, 'val_new_sgm.npy'), val_acc_new)
    np.save(os.path.join(exp_dir, 'val_base_sgm.npy'), val_acc_base)
    print('\nRuntime for entire experiment (in mins): %0.3f' % ((time.time() - start_time)/60))

    ### /// END /// ###


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Directories and names
    parser.add_argument('--expt_name', type=str)  # name of the experiment
    parser.add_argument('--label_dir', type=str, default=None)  # directory for numpy label files
    parser.add_argument('--images_dir', type=str, default=None)  # directory for Places365 train/val folders
    parser.add_argument('--label_dir_old', type=str, default=None)  # directory for numpy label files
    parser.add_argument('--images_dir_old', type=str, default=None)  # directory for ImageNet1K train/val folders
    parser.add_argument('--save_dir', type=str, required=False)  # directory for saving results
    parser.add_argument('--resume_full_path', type=str, default=None)  # directory of previous model to load
    parser.add_argument('--ckpt_file', type=str, default='pretrained_checkpoint.pth')
    # network parameters
    parser.add_argument('--base_arch', type=str, default='ConvNeXtNet')  # architecture for G
    parser.add_argument('--classifier', type=str, default='ConvNeXt_block2')  # architecture for F
    parser.add_argument('--hidden_dim', type=int, default=384) ## hidden dim in penulitmate layer
    parser.add_argument('--classifier_ckpt', type=str, required=True)  # base-init ckpt ///// Pretrain Checkpoint /////
    parser.add_argument('--extract_features_from', type=str,
                        default='model.downsample_layers.2')  # name of the layer to extract features
    # training params
    parser.add_argument('--weight_decay', type=float, default=5e-2)  # weight decay for network
    parser.add_argument('--batch_size', type=int, default=128) # dataloader batch size (# of rehearsal samples in a minibatch)
    parser.add_argument('--num_iter', type=int, default=1200)  # total number of training iterations
    parser.add_argument('--lr', type=float, default=0.2)  # starting lr for CL training
    # replay buffer parameters
    parser.add_argument('--max_buffer_size', type=int, default=None)  # maximum number of samples in buffer
    # CL Setup
    parser.add_argument('--num_classes', type=int, default=365)  # total number of classes
    parser.add_argument('--min_class', type=int, default=0)  # overall minimum class
    parser.add_argument('--base_init_classes', type=int, default=1000)  # number of base init classes
    parser.add_argument('--class_increment', type=int, default=73)  # how often to evaluate
    parser.add_argument('--cl_min_class', type=int, default=0)  # class to begin continual training
    parser.add_argument('--cl_max_class', type=int, default=365)  # class to end continual training
    parser.add_argument('--seed', type=int, default=1993)

    # Get arguments and print them out and make any necessary directories
    args = parser.parse_args()

    wandb.init(name='SGM_CL_' + str(args.seed),
    project="ConvNeXt", entity="user")
    wandb.config = {
    "learning_rate": args.lr,
    "batch_size": args.batch_size
    }

    if args.save_dir is None:
        args.save_dir = 'CL_experiments/' + args.expt_name

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print("Arguments {}".format(json.dumps(vars(args), indent=4, sort_keys=True)))

    # Make model and begin continual learning
    sgm = SModel(num_classes=args.num_classes, classifier_G=args.base_arch,
                extract_features_from=args.extract_features_from, classifier_F=args.classifier,
                hidden_dim=args.hidden_dim, classifier_ckpt=args.classifier_ckpt,
                weight_decay=args.weight_decay, base_init_classes=args.base_init_classes,
                class_increment=args.class_increment, max_buffer_size=args.max_buffer_size,
                lr=args.lr, seed=args.seed)

    ## Continual Learning
    continual_learning(args, sgm, wandb)
