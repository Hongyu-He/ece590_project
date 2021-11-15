import copy
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from sampling import sample_iid, sample_noniid
from pytorchyolo.utils.datasets import ListDataset
from pycocotools.coco import COCO
from tqdm import tqdm
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS
from pytorchyolo.utils.utils import to_cpu, load_classes, print_environment_info, provide_determinism, worker_seed_set


def id2name(coco):
    classes=dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']]=cls['name']
    return classes


def _create_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_training=False):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader, dataset


def dataset_generator(server_num, batch_size, n_cpu=8, multiscale_training=False, 
                      img_size=416, iid=True, mode="train"):
    """Generate dataset and dataloader for each local server.

    Args:
        server_num ([type]): [description]
        batch_size ([type]): [description]
        n_cpu (int, optional): [description]. Defaults to 8.
        multiscale_training (bool, optional): [description]. Defaults to False.
        img_size (int, optional): [description]. Defaults to 416.
        iid (bool, optional): [description]. Defaults to True.
        mode (str, optional): [description]. Defaults to "train".

    Returns:
        [type]: [description]
    """
    dataDir = '/home/hh239/ece590/ece590_project/YOLOv3/data/coco'
    dataset = mode + '2014'
    prefix = '/home/hh239/ece590/ece590_project/data_pointer/'
    Path(prefix).mkdir(parents=True, exist_ok=True)
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataset)
    classes_names = ['car', 'bicycle', 'person', 'motorcycle', 'bus', 'truck']
    coco = COCO(annFile)
    classes = id2name(coco)
    print(classes)
    classes_ids = coco.getCatIds(catNms=classes_names)
    print(classes_ids)
    if iid is True:
        text = dict()
        for i_s in range(server_num):
            file_out = prefix + dataset + 'server' + '_' + str(i_s) + '_' + '.txt'
            f = open(file_out, 'a')
            for cls in classes_names:
                #Get ID number of this class
                cls_id = coco.getCatIds(catNms=[cls])
                img_ids = coco.getImgIds(catIds=cls_id)
                total = len(img_ids)
                per = int(total/server_num)
                img_ids_ = img_ids[(i_s)*per:(i_s+1)*per]

                for imgId in tqdm(img_ids_):
                    img = coco.loadImgs(imgId)[0]
                    filename = img['file_name']
                    string = dataDir + '/images/' + dataset + '/' + filename
                    #print(string)
                    f.write(string)
                    f.write('\n')
            f.close()
            f = open(file_out, "r")
            
            
            writeDir = prefix + "new_" + dataset + 'server' + '_' + str(i_s) + '_' + ".txt"
            outfile=open(writeDir,"w")
            lines_seen = set()  # Build an unordered collection of unique elements.
 
            for line in f:
                line = line.strip('\n')
                if line not in lines_seen:
                    outfile.write(line+ '\n')
                    lines_seen.add(line)
            outfile.close()
            text[i_s] = writeDir 
            
        dataloader_ser = {}
        dataset_ser = {}
    
        for i_s in range(server_num):
            img_path = text[i_s]
            dataloader, dataset =  _create_data_loader(img_path, batch_size, 
                                                       img_size=img_size, n_cpu=n_cpu, 
                                                       multiscale_training=multiscale_training)
            dataloader_ser[i_s] = dataloader
            dataset_ser[i_s] = dataset
        print('done')
        
        return dataloader_ser, dataset_ser


# def get_dataset(dataset, num_users, iid=True):
#     """Returns train and test datasets and a user group which is a dict where
#     the keys are the user index and the values are the corresponding data for
#     each of those users.

#     Args:
#         dataset ([type]): [description]
#         num_users ([type]): [description]
#         iid (bool, optional): [description]. Defaults to True.

#     Returns:
#         [type]: [description]
#     """
#     data_dir = '../data/chest_xray/'
#     # TODO try other transformers, use different transformer for train/test/val
#     apply_transform = transforms.Compose(
#         [transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#     trainset = ImageFolder(root=data_dir+'train', transform=apply_transform)
#     testset = ImageFolder(root=data_dir+'test', transform=apply_transform)
#     valset = ImageFolder(root=data_dir+'val', transform=apply_transform)

#     # sample training data amongst users
#     if iid:
#         # Sample IID user data
#         user_groups = sample_iid(trainset, num_users)
#     else:
#         # TODO Sample Non-IID user data
#         # user_groups = sample_noniid(trainset, num_users)
#         user_groups = {}
#         raise NotImplementedError("Can't sample Non-IID user data")

#     return trainset, testset, valset, user_groups


def sample_imgs(dataset, num, plot=True):
    """Sample and plot images from dataset.

    Args:
        dataset ([type]): [description]
        num ([type]): [description]
        plot (bool, optional): [description]. Defaults to True.
    """
    # TODO implement me


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
