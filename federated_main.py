import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from pytorchyolo.models import load_model
from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import dataset_generator, average_weights, exp_details
from pytorchyolo.test import _evaluate
from beautifultable import BeautifulTable
import pickle


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')

    args = args_parser()
    exp_details(args)  # print experiment details

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset and user groups
    train_loader, train_set = dataset_generator(server_num=args.num_users,
                                                batch_size=args.batch_size,
                                                iid=True,
                                                mode='train')
    test_loader, test_set = dataset_generator(server_num=args.num_users,
                                              batch_size=args.batch_size,
                                              iid=True,
                                              mode='val')

    # BUILD MODEL
    global_model = load_model(args.model, args.pretrained_weights)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    print_every = 2

    metric_local = {}
    metric_global = {}
    weights_local_record = {}
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        metric_local[epoch] = {}
        weights_local_record[epoch] = {}
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        # pick up clients that participate in this round 
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, trainloader=train_loader[idx],
                                      testloader=test_loader[idx], idx=idx)
            # train local models for local_ep epochs
            w, loss, local_eval = local_model.update_weights(model=copy.deepcopy(global_model))
            metric_local[epoch][idx] = local_eval
            weights_local_record[epoch][idx] = copy.deepcopy(w)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # evaluate global model
        metric_global[epoch] = {}
        for i in test_loader:
            print(f"\n---- Evaluating Global Model ----\n")
            name_file = open("/home/hh239/ece590/ece590_project/YOLOv3/data/coco.names", 'r')
            class_names = name_file.read().split('\n').remove('')
            name_file.close()
            metrics_output = _evaluate(
                global_model,
                test_loader[i],
                class_names,
                img_size=global_model.hyperparams['height'],
                iou_thres=args.iou_thres,
                conf_thres=args.conf_thres,
                nms_thres=args.nms_thres,
                verbose=args.verbose
            )
            metric_global[epoch][i] = metrics_output
            if metrics_output is not None:
                print(f"\n---- Evaluating Matrics on testset {i} ----\n")
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = {
                    "precision": precision.mean(),
                    "recall": recall.mean(),
                    "mAP": AP.mean(),
                    "f1": f1.mean()}
                eval_table = BeautifulTable()
                eval_table.rows.append(["precision", precision.mean()])
                eval_table.rows.append(["recall", recall.mean()])
                eval_table.rows.append(["mAP", AP.mean()])
                eval_table.rows.append(["f1", f1.mean()])
                print(eval_table)
    
    results_path = "/home/hh239/ece590/ece590_project/results"
    Path(results_path).mkdir(parents=True, exist_ok=True)
    ckp_path = results_path + '/checkpoints'
    Path(ckp_path).mkdir(parents=True, exist_ok=True)
    torch.save(global_model.state_dict(), ckp_path+'/ckp')
    with open(results_path+'/metric_local.pkl', 'wb') as f:
        pickle.dump(metric_local)
    with open(results_path+'/metric_global.pkl', 'wb') as f:
        pickle.dump(metric_global)
    with open(results_path+'/weights_local.pkl', 'wb') as f:
        pickle.dump(weights_local_record)

    # Test inference after completion of training
    # test_acc, test_loss = test_inference(args, global_model, test_dataset)

    # print(f' \n Results after {args.epochs} global rounds of training:')
    # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    # print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # # Saving the objects train_loss and train_accuracy:
    # file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)

    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)

    # print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
