import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.utils.utils import to_cpu
from pytorchyolo.test import _evaluate
from terminaltables import AsciiTable
from beautifultable import BeautifulTable


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, trainloader, testloader, idx):
        self.args = args
        self.idx = idx  # client index
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def update_weights(self, model, eval=True):
        # Set mode to train model
        model.train()
        epoch_loss = []  # average loss for each epoch
        print(f"\n---- Training Client Model {self.idx} ----")

        # Set optimizer for the local updates
        params = [p for p in model.parameters() if p.requires_grad]

        if (model.hyperparams['optimizer'] in [None, "adam"]):
            optimizer = optim.Adam(
                params,
                lr=model.hyperparams['learning_rate'],
                weight_decay=model.hyperparams['decay'],
            )
        elif (model.hyperparams['optimizer'] == "sgd"):
            optimizer = optim.SGD(
                params,
                lr=model.hyperparams['learning_rate'],
                weight_decay=model.hyperparams['decay'],
                momentum=model.hyperparams['momentum'])
        else:
            print("Unknown optimizer. Please choose between (adam, sgd).")    

        for iter in range(self.args.local_ep):
            batch_loss = []  # average loss for each batch
            for batch_i, (_, imgs, targets) in enumerate(tqdm(self.trainloader, 
                                                desc=f"Training Epoch {iter}/{self.args.local_ep}")):
                batches_done = len(self.trainloader) * iter + batch_i

                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device)

                outputs = model(imgs)

                loss, loss_components = compute_loss(outputs, targets, model)
                loss.backward()

                # Run optimizer
                if batches_done % model.hyperparams['subdivisions'] == 0:
                    # Adapt learning rate
                    # Get learning rate defined in cfg
                    lr = model.hyperparams['learning_rate']
                    if batches_done < model.hyperparams['burn_in']:
                        # Burn in
                        lr *= (batches_done / model.hyperparams['burn_in'])
                    else:
                        # Set and parse the learning rate to the steps defined in the cfg
                        for threshold, value in model.hyperparams['lr_steps']:
                            if batches_done > threshold:
                                lr *= value
                    
                    # Set learning rate
                    for g in optimizer.param_groups:
                        g['lr'] = lr

                    # Run optimizer
                    optimizer.step()
                    # Reset gradients
                    optimizer.zero_grad() 

                # if self.args.verbose:
                #     print(AsciiTable(
                #         [
                #             ["Type", "Value"],
                #             ["IoU loss", float(loss_components[0])],
                #             ["Object loss", float(loss_components[1])],
                #             ["Class loss", float(loss_components[2])],
                #             ["Loss", float(loss_components[3])],
                #             ["Batch loss", to_cpu(loss).item()],
                #         ]).table)
                batch_loss.append(to_cpu(loss).item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        # evaluate local model
        print(f"\n---- Evaluating Local Model {self.idx} ----")
        name_file = open("/home/hh239/ece590/ece590_project/YOLOv3/data/coco.names", 'r')
        class_names = name_file.read().split('\n').remove('')
        name_file.close()
        metrics_output = _evaluate(
            model,
            self.testloader,
            class_names,
            img_size=model.hyperparams['height'],
            iou_thres=self.args.iou_thres,
            conf_thres=self.args.conf_thres,
            nms_thres=self.args.nms_thres,
            verbose=self.args.verbose
        )
        if metrics_output is not None:
            print(f"\n---- Evaluating Matrics ----\n")
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

        # local model weights and average epoch loss are returned
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), evaluation_metrics

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            # batch_loss = self.criterion(outputs, labels)
            # TODO implement loss
            batch_loss = compute_loss(...)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
