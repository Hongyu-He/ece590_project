import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.utils.utils import to_cpu
from terminaltables import AsciiTable


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

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model):
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

                if self.args.verbose:
                    print(AsciiTable(
                        [
                            ["Type", "Value"],
                            ["IoU loss", float(loss_components[0])],
                            ["Object loss", float(loss_components[1])],
                            ["Class loss", float(loss_components[2])],
                            ["Loss", float(loss_components[3])],
                            ["Batch loss", to_cpu(loss).item()],
                        ]).table)
                batch_loss.append(to_cpu(loss).item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

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
