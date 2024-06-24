import re
import glob
from models import *
import omegaconf
import argparse
# from easy_receptive_fields_pytorch.receptivefield import receptivefield, give_effective_receptive_field
import copy
import math
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
# import clip
# import wandb
import typing
import os
import torch.nn.functional as F
from torchvision import transforms as trnfs
import time
# from dataset import ImageNet
from pathlib import Path
# from torchvision.models import resnet18, ResNet18_Weights, \
#     resnet34, ResNet34_Weights, \
#     resnet50, ResNet50_Weights, \
#     resnet152, ResNet152_Weights, \
#     mobilenet_v3_large, MobileNet_V3_Large_Weights, vit_b_32, ViT_B_32_Weights, \
#     efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import resnet34, mobilenet_v3_large, efficientnet_b0, ResNet34_Weights
import torchvision
# import vits
import timm
# from torchsummary import summary
import pandas as pd
from train_CIFAR10 import get_datasets
import math

imagenet_normalize = trnfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalize = trnfs.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
print("Imported everything")


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, feature_dim, num_classes, normalize=False, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.classification_head = torch.nn.Linear(feature_dim, num_classes)
        self.normalize = normalize
        if initial_weights is None:
            initial_weights = torch.zeros_like(self.classification_head.weight)
            torch.nn.init.kaiming_uniform_(initial_weights, a=math.sqrt(5))
        self.classification_head.weight = torch.nn.Parameter(initial_weights.clone())
        self.classification_head.bias = torch.nn.Parameter(
            torch.zeros_like(self.classification_head.bias))

        # Note: modified. Get rid of the language part.
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images, return_features=False):
        features = self.model.encode_image(images)
        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        logits = self.classification_head(features)
        if return_features:
            return logits, features
        return logits


class CustomValImageNetDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading training/validation images and labels from directory and applying transformations.

        Attributes:
        root_dir (str): dataset root directory
        transform (Compose): transformations that are to be applied to the image.
        samples (List[Tuple[str, int]]): list of tuples containing the path to an image and class index.
        class_map (dict): dictionary mapping class index to its str class name.

    Args:
        root_dir (str): root directory of the dataset, without the "train_set/train_set" directory from the zip file.
        class_dir (str): root directory of the text file containing class index to class name mappings. Defaults to "class.txt".
        transform (Compose, optional): transformations that are to be applied to the images. Defaults to None.
    """

    def __init__(self, root_dir="/jmain02/flash/share/datasets/ImageNet/ILSVRC2012/ValidationSet",
                 ground_truth_file="/jmain02/home/J2AD014/mtc03/lla98-mtc03/imagenet_val/ILSVRC2015_clsloc_validation_ground_truth.txt",
                 transform=None):

        """
        Initialises dataset object: sets up directory paths, loads class mappings.

        Parameters:
            root_dir (str): root directory of the dataset, without the "train_set/train_set" directory from the zip file.
            class_dir (str): root directory of the text file containing class index to class name mappings. Defaults to "class.txt".
            transform (Compose, optional): transformations that are to be applied to the images. Defaults to None.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_map = {}

        # Load class mappings within initializer
        #        : program this so it works in JADE with val directory in /jmain02/flash/share/datasets/ImageNet/ILSVRC2012/ValidationSet  and class file in  /jmain02/flash/share/datasets/ImageNet/ILSVRC2012/DevKit/data/ILSVRC2012_validation_ground_truth.txt
        #
        val_classes_ground_truth = np.loadtxt(ground_truth_file)
        val_classes_ground_truth = val_classes_ground_truth - 1
        # with open(os.path.join(root_dir), "r") as f:
        #
        #     for line in f:
        #         index, class_dir = line.strip().split("\t")
        #
        #         self.class_map[int(index)] = class_dir

        # Populate the samples list with tuples of image file paths and their corresponding class indices for all classes.
        # for class_index, class_dir in self.class_map.items():
        #     class= os.path.join(self.root_dir, class_dir)

        for i, img_file in enumerate(os.listdir(self.root_dir)):
            image_path = os.path.join(self.root_dir, img_file)
            self.samples.append((image_path, val_classes_ground_truth[i]))

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: number of samples
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns an image and its corresponding label from the dataset at the specified index. Optionally, applies a series of transformations.

        Parameters:
            idx (int): index of retrieved sample.

        Returns:
            two-element tuple:
                1) transformed (if applicable) image tensor
                2) Int label of image class
        """
        img_path, class_index = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, class_index


def get_arc3_dataset(cfg, transforms=None):
    # Excerpt take from https://github.com/pytorch/examples/blob/e0d33a69bec3eb4096c265451dbb85975eb961ea/imagenet/main.py#L113-L126
    # Data loading code

    current_directory = Path().cwd()
    data_path = ""
    if "sclaam" == current_directory.owner() or "sclaam" in current_directory.__str__():
        data_path = "/nobackup/sclaam/data/"
    elif "luis alfredo" == current_directory.owner() or "luis alfredo" in current_directory.__str__():
        data_path = "c:/users\luis alfredo\onedrive - university of leeds\phd\datasets\mnist"
    elif "luisaam" == current_directory.owner() or "luisaam" in current_directory.__str__():
        data_path = "datasets/"
    traindir = data_path + 'imagenet/' + 'train'
    testdir = data_path + 'imagenet/' + 'val'

    if transforms:

        train_trasnform = transforms["train"]
        test_trasnform = transforms["val"]

    else:

        train_transform = trnfs.Compose([
            trnfs.RandomResizedCrop(224),
            trnfs.RandomHorizontalFlip(),
            trnfs.ToTensor(),
            normalize,
        ])
        test_transform = trnfs.Compose([
            trnfs.Resize(256),
            trnfs.CenterCrop(224),
            trnfs.ToTensor(),
            normalize,
        ])
    whole_train_dataset = torchvision.datasets.ImageFolder(
        traindir, train_transform)
    print(f"Length of dataset: {len(whole_train_dataset)}")

    train_dataset, val_dataset = torch.utils.data.random_split(whole_train_dataset, [1231167, 50000])

    full_test_dataset = torchvision.datasets.ImageFolder(testdir, test_transform)
    print(full_test_dataset.imgs)

    big_test, small_test = torch.utils.data.random_split(full_test_dataset, [len(full_test_dataset) - 10000, 10000])

    # This code is to transform it into the "fast" format of ffcv

    # my_dataset = val_dataset
    # write_path = data_path + "imagenet/valSplit_dataset.beton"

    # For the validation set that I use to recover accuracy

    # # Pass a type for each data field
    # writer = DatasetWriter(write_path, {
    #     # Tune options to optimize dataset size, throughput at train-time
    #     'image': RGBImageField(
    #         max_resolution=256,
    #         jpeg_quality=90
    #     ),
    #     'label': IntField()
    # })
    # # Write dataset
    # writer.from_indexed_dataset(my_dataset)

    # For the validation set that I use to recover accuracy

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.workers, pin_memory=True, sampler=None)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.workers, pin_memory=True, sampler=None)
    # if cfg.length_test == "small":
    #     test_loader = torch.utils.data.DataLoader(
    #         small_test,
    #         batch_size=cfg.batch_size, shuffle=False,
    #         num_workers=cfg.workers, pin_memory=True)
    # if cfg.length_test == "big":
    #     test_loader = torch.utils.data.DataLoader(
    #         big_test,
    #         batch_size=cfg.batch_size, shuffle=False,
    #         num_workers=cfg.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        full_test_dataset,
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def remove_reparametrization(model, name_module="", exclude_layer_list: list = []):
    for name, m in model.named_modules():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d) and name not in exclude_layer_list:
            if name_module == "":
                prune.remove(m, "weight")
            if name == name_module:
                prune.remove(m, "weight")
                break


def get_layer_dict(model: torch.nn.Module):
    """
    :param model:
    :return: The name of the modules that are not batch norm and their correspondent weight with original shape.
    """
    iter_1 = model.named_modules()
    layer_dict = []

    for name, m in iter_1:
        with torch.no_grad():
            if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not \
                    isinstance(m, nn.BatchNorm3d):
                layer_dict.append((name, m.weight.data.cpu().detach()))
    #
    return layer_dict


def prepare_val_imagenet(args):
    train_transform = trnfs.Compose([
        trnfs.RandomResizedCrop(224),
        trnfs.RandomHorizontalFlip(),
        trnfs.ToTensor(),
        imagenet_normalize,
    ])
    test_transform = trnfs.Compose([
        trnfs.Resize(256),
        trnfs.CenterCrop(224),
        trnfs.ToTensor(),
        imagenet_normalize,
    ])
    val_dataset = CustomValImageNetDataset(transform=test_transform)

    test_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    return test_loader


def is_prunable_module(m: torch.nn.Module):
    return (isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('/datasets'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--experiment",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="resnet34",
    )
    return parser.parse_args()


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_zero_parameters(model):
    return sum((p == 0).sum() for p in model.parameters() if p.requires_grad)


def get_model_from_sd(state_dict, base_model):
    feature_dim = state_dict['classification_head.weight'].shape[1]
    num_classes = state_dict['classification_head.weight'].shape[0]
    model = ModelWrapper(base_model, feature_dim, num_classes, normalize=True)
    for p in model.parameters():
        p.data = p.data.float()
    model.load_state_dict(state_dict)
    model = model.cuda()
    devices = [x for x in range(torch.cuda.device_count())]
    return torch.nn.DataParallel(model, device_ids=devices)


def create_feature_extractor(model, level=4):
    el_children = list(model.children())
    if level == 4:

        # resnet34
        if isinstance(model, type(resnet34())):
            last_index = -2
            feature_extractor = torch.nn.Sequential(*el_children[:last_index])
            return feature_extractor

        # legacy_seresnet34.in1k

        if isinstance(model, type(timm.create_model('legacy_seresnet34.in1k'))):
            last_index = -2
            feature_extractor = torch.nn.Sequential(*el_children[:last_index])
            return feature_extractor

        # SK-ResNet-34

        if isinstance(model, type(timm.create_model('skresnet34'))):
            last_index = -2
            feature_extractor = torch.nn.Sequential(*el_children[:last_index])
            return feature_extractor

        # Is mobilenetv2
        if isinstance(model, type(timm.create_model('mobilenetv2_120d'))):
            last_index = -2
            feature_extractor = torch.nn.Sequential(*el_children[:last_index])
            return feature_extractor
        if isinstance(model, type(mobilenet_v3_large())):
            feature_extractor = el_children[0]
            return feature_extractor
        if isinstance(model, type(efficientnet_b0())):
            feature_extractor = el_children[0]
            return feature_extractor


    elif level == 3:

        # resnet34
        if isinstance(model, type(resnet34())):
            last_index = 7
            feature_extractor = torch.nn.Sequential(*el_children[:last_index])
            return feature_extractor

        # legacy_seresnet34.in1k
        if isinstance(model, type(timm.create_model('legacy_seresnet34.in1k'))):
            last_index = 5
            feature_extractor = torch.nn.Sequential(*el_children[:last_index])
            return feature_extractor

        # SK-ResNet-34

        if isinstance(model, type(timm.create_model('skresnet34'))):
            last_index = 7
            feature_extractor = torch.nn.Sequential(*el_children[:last_index])
            return feature_extractor

        # Is mobilenetv2
        if isinstance(model, type(timm.create_model('mobilenetv2_120d'))):
            temp = []
            temp2 = copy.deepcopy(el_children[-5])
            temp.extend(el_children[:2])
            temp.append(temp2[:4])

            feature_extractor = torch.nn.Sequential(*temp)

            return feature_extractor

        # Is mobilenetv3
        if isinstance(model, type(mobilenet_v3_large())):
            full_extractor = list(el_children[0])
            feature_extractor = torch.nn.Sequential(*full_extractor[:10])
            return feature_extractor
        # Is efficientNet-B0
        if isinstance(model, type(efficientnet_b0())):
            full_extractor = list(el_children[0])
            feature_extractor = torch.nn.Sequential(*full_extractor[:5])
            return feature_extractor



    elif level == 2:

        # resnet34
        if isinstance(model, type(resnet34())):
            last_index = 5
            feature_extractor = torch.nn.Sequential(*el_children[:last_index])
            return feature_extractor

        # legacy_seresnet34.in1k
        if isinstance(model, type(timm.create_model('legacy_seresnet34.in1k'))):
            last_index = 4
            feature_extractor = torch.nn.Sequential(*el_children[:last_index])
            return feature_extractor

        # SK-ResNet-34

        if isinstance(model, type(timm.create_model('skresnet34'))):
            last_index = 5
            feature_extractor = torch.nn.Sequential(*el_children[:last_index])
            return feature_extractor

        # Is mobilenetv2

        if isinstance(model, type(timm.create_model('mobilenetv2_120d'))):
            temp = []
            temp2 = copy.deepcopy(el_children[-5])
            temp.extend(el_children[:2])
            temp.append(temp2[0])

            feature_extractor = torch.nn.Sequential(*temp)

            return feature_extractor

        if isinstance(model, type(mobilenet_v3_large())):
            full_extractor = list(el_children[0])
            feature_extractor = torch.nn.Sequential(*full_extractor[:6])
            return feature_extractor

        if isinstance(model, type(efficientnet_b0())):
            full_extractor = list(el_children[0])
            feature_extractor = torch.nn.Sequential(*full_extractor[:2])
            return feature_extractor


    elif level == 1:

        # resnet34
        if isinstance(model, type(resnet34())):
            last_index = 4
            feature_extractor = torch.nn.Sequential(*el_children[:last_index])
            return feature_extractor

        # legacy_seresnet34.in1k
        if isinstance(model, type(timm.create_model('legacy_seresnet34.in1k'))):
            last_index = 1
            feature_extractor = torch.nn.Sequential(*el_children[:last_index])
            return feature_extractor

        # SK-ResNet-34
        if isinstance(model, type(timm.create_model('skresnet34'))):
            last_index = 4
            feature_extractor = torch.nn.Sequential(*el_children[:last_index])
            return feature_extractor

        # Is mobilenetv2
        if isinstance(model, type(timm.create_model('mobilenetv2_120d'))):
            last_index = -6
            feature_extractor = torch.nn.Sequential(*el_children[:last_index])
            return feature_extractor

        # Is mobilenetv3

        if isinstance(model, type(mobilenet_v3_large())):
            feature_extractor = el_children[0][0]
            return feature_extractor

        # Is EfficientNet-B0
        if isinstance(model, type(efficientnet_b0())):
            feature_extractor = el_children[0][0]
            return feature_extractor

    else:
        last_index = -2

    # feature_extractor = torch.nn.Sequential(*el_children[:last_index])
    #
    # return feature_extractor


def check_if_has_encode_image(model):
    invert_op = getattr(model, "invert_op", None)
    if callable(invert_op):
        invert_op(model.path.parent_op)


def prune_function(net, cfg, pr_per_layer=None):
    target_sparsity = cfg.amount
    if cfg.pruner == "global":
        prune_with_rate(net, target_sparsity, exclude_layers=cfg.exclude_layers, type="global")
    if cfg.pruner == "manual":
        prune_with_rate(net, target_sparsity, exclude_layers=cfg.exclude_layers, type="layer-wise",
                        pruner="manual", pr_per_layer=pr_per_layer)
        individual_prs_per_layer = prune_with_rate(net, target_sparsity,
                                                   exclude_layers=cfg.exclude_layers, type="layer-wise",
                                                   pruner="lamp", return_pr_per_layer=True)
        if cfg.use_wandb:
            log_dict = {}
            for name, elem in individual_prs_per_layer.items():
                log_dict["individual_{}_pr".format(name)] = elem
            wandb.log(log_dict)
    if cfg.pruner == "lamp":
        prune_with_rate(net, target_sparsity, exclude_layers=cfg.exclude_layers,
                        type="layer-wise",
                        pruner=cfg.pruner)


def weights_to_prune(model: torch.nn.Module, exclude_layer_list=[]):
    modules = []
    for name, m in model.named_modules():
        if hasattr(m, 'weight') and type(m) != nn.BatchNorm1d and not isinstance(m, nn.BatchNorm2d) and not isinstance(
                m, nn.BatchNorm3d) and name not in exclude_layer_list:
            modules.append((m, "weight"))
            # print(name)

    return modules


def prune_with_rate(net: torch.nn.Module, amount: typing.Union[int, float], pruner: str = "erk",
                    type: str = "global",
                    criterion:
                    str =
                    "l1", exclude_layers: list = [], pr_per_layer: dict = {}, return_pr_per_layer: bool = False,
                    is_stochastic: bool = False, noise_type: str = "", noise_amplitude=0):
    if type == "global":
        print("Exclude layers in prun_with_rate:{}".format(exclude_layers))
        weights = weights_to_prune(net, exclude_layer_list=exclude_layers)
        print("Length of weigths to prune:{}".format(len(weights))
              )
        if criterion == "l1":
            prune.global_unstructured(
                weights,
                pruning_method=prune.L1Unstructured,
                amount=amount
            )
        if criterion == "l2":
            prune.global_unstructured(
                weights,
                pruning_method=prune.LnStructured,
                amount=amount,
                n=2
            )
    elif type == "layer-wise":
        from layer_adaptive_sparsity.tools.pruners import weight_pruner_loader
        if pruner == "lamp":
            pruner = weight_pruner_loader(pruner)
            if return_pr_per_layer:
                return pruner(model=net, amount=amount, exclude_layers=exclude_layers,
                              return_amounts=return_pr_per_layer)
            else:
                pruner(model=net, amount=amount, exclude_layers=exclude_layers, is_stochastic=is_stochastic,
                       noise_type=noise_type, noise_amplitude=noise_amplitude)
        if pruner == "erk":
            pruner = weight_pruner_loader(pruner)
            pruner(model=net, amount=amount, exclude_layers=exclude_layers)
            # _, amount_per_layer, _, _ = erdos_renyi_per_layer_pruning_rate(model=net, cfg=cfg)
            # names, weights = zip(*get_layer_dict(net))
            # for name, module in net.named_modules():
            #     if name in exclude_layers or name not in names:
            #         continue
            #     else:
            #         prune.l1_unstructured(module, name="weight", amount=float(amount_per_layer[name]))
        if pruner == "manual":
            for name, module in net.named_modules():
                with torch.no_grad():
                    if name in exclude_layers or not is_prunable_module(module):
                        continue
                    else:
                        prune.l1_unstructured(module, name="weight", amount=float(pr_per_layer[name]))
    elif type == "random":
        weights = weights_to_prune(net, exclude_layer_list=exclude_layers)
        if criterion == "l1":
            prune.random_structured(
                weights,
                # pruning_method=prune.L1Unstructured,
                amount=amount
            )


    else:
        raise NotImplementedError("Not implemented for type {}".format(type))


def test(net, use_cuda, testloader, one_batch=False, verbose=2, count_flops=False, batch_flops=0, number_batches=0):
    if use_cuda:
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    if count_flops:
        assert batch_flops != 0, "If count_flops is True,batch_flops must be non-zero"

    sparse_flops = 0

    first_time = 1

    sparse_flops_batch = 0
    with torch.no_grad():
        # print("Before the dataloader loop")
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # if batch_idx == 0:
            # print("In the data loader loop")
            if use_cuda:
                targets = targets.type(torch.LongTensor)
                inputs, targets = inputs.cuda(), targets.cuda()
            # if batch_idx == 0:
            # print("before forward method")
            outputs = net(inputs)
            # if batch_idx == 0:
            # print("batch indx {}".format(batch_idx))
            # print("After the forward method")
            # print("outputs.size: {}".format(outputs.size()))
            # print("targets.size: {}".format(targets.size()))
            loss = criterion(outputs, targets)
            if count_flops:
                sparse_flops += batch_flops
            # print("After loss calculation!")
            test_loss += loss.data.item()

            # if torch.all(outputs > 0):

            _, predicted = torch.max(outputs.data, 1)

            top5_probabilities, top5_class_indices = torch.topk(outputs.softmax(dim=1) * 100, k=5)

            top1_probabilities, top1_class_indices = torch.topk(outputs.softmax(dim=1) * 100, k=1)

            if batch_idx == 0:
                print("Top 1 class index: {}\n          Targets: {}".format(top1_class_indices.data.cpu(),
                                                                            targets.data.cpu()))

                print("\n")

                print("Top 5 class index: {}\n          Targets: {}".format(top5_class_indices.data.cpu(),
                                                                            targets.data.cpu()))

                print("Predicted by my function: {}\n         Targets: {}".format(predicted.data.cpu(),
                                                                                  targets.data.cpu()))

                break

            # else:
            #
            #     soft_max_outputs = F.softmax(outputs, dim=1)
            #
            #     _, predicted = torch.max(soft_max_outputs, 1)

            total += targets.size(0)

            correct += predicted.eq(targets.data).cpu().sum()

            # print(correct/total)

            if batch_idx % 100 == 0:
                if verbose == 2:
                    print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                          % (test_loss / (batch_idx + 1), 100. * correct.item() / total, correct, total))
            if one_batch:
                if count_flops:
                    return 100. * correct.item() / total, sparse_flops
                else:
                    return 100. * correct.item() / total

            if number_batches > 0:
                if number_batches < batch_idx:
                    return 100. * correct.item() / total

    if verbose == 1 or verbose == 2:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss / len(testloader), correct, total,
            100. * correct.item() / total))
    # net.cpu()
    if count_flops:
        return 100. * correct.item() / total, sparse_flops
    else:
        return 100. * correct.item() / total


def main(args):
    if args.model == "vgg19":
        exclude_layers = ["features.0", "classifier"]
    else:
        exclude_layers = ["conv1", "linear"]

    cfg = omegaconf.DictConfig(
        {"architecture": args.model,
         "model_type": "alternative",
         # "model_type": "hub",
         "solution": "trained_models/cifar10/resnet50_cifar10.pth",
         # "solution": "trained_m
         "dataset": args.dataset,
         "batch_size": 128,
         "num_workers": args.num_workers,
         "amount": args.pruning_rate,
         "noise": "gaussian",
         "sigma": 0.005,
         "pruner": "global",
         "exclude_layers": exclude_layers

         })

    if args.ffcv:
        from ffcv_loaders import make_ffcv_small_imagenet_dataloaders
        train, _, testloader = make_ffcv_small_imagenet_dataloaders(args.ffcv_train, args.ffcv_val,
                                                                    128, args.num_workers)
    else:
        train, val, testloader = get_datasets(args)

    from torchvision.models import resnet18, resnet50

    if args.model == "resnet18":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet18_rf(num_classes=10, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet18_rf(num_classes=100, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet18_rf(num_classes=200, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = ResNet18_rf(num_classes=200, RF_level=args.RF_level)
    if args.model == "resnet50":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet50_rf(num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet50_rf(num_classes=100, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet50_rf(num_classes=200, rf_level=args.RF_level)
        if args.type == "pytorch" and args.dataset == "cifar10":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)
    if args.model == "vgg19":
        if args.type == "normal" and args.dataset == "cifar10":
            net = VGG_RF("VGG19_rf", num_classes=10, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = VGG_RF("VGG19_rf", num_classes=100, RF_level=args.RF_level)

        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = VGG_RF("VGG19_rf", num_classes=200, RF_level=args.RF_level)
    if args.model == "resnet24":
        if args.type == "normal" and args.dataset == "cifar10":
            net = ResNet24_rf(num_classes=10, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "cifar100":
            net = ResNet24_rf(num_classes=100, rf_level=args.RF_level)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = ResNet24_rf(num_classes=200, rf_level=args.RF_level)
        if args.type == "pytorch" and args.dataset == "cifar10":
            # # net = resnet50()
            # # in_features = net.fc.in_features
            # net.fc = nn.Linear(in_features, 10)
            raise NotImplementedError(
                " There is no implementation for this combination {}, {} {} ".format(args.model, args.type))
    if args.model == "resnet_small":

        if args.type == "normal" and args.dataset == "cifar10":
            net = small_ResNet_rf(num_classes=10, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "cifar100":
            net = small_ResNet_rf(num_classes=100, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "tiny_imagenet":
            net = small_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "normal" and args.dataset == "small_imagenet":
            net = small_ResNet_rf(num_classes=200, RF_level=args.RF_level, multiplier=args.width)
        if args.type == "pytorch" and args.dataset == "cifar10":
            raise NotImplementedError
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 10)
        if args.type == "pytorch" and args.dataset == "cifar100":
            raise NotImplementedError
            net = resnet50()
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, 100)

    dense_accuracy_list = []
    pruned_accuracy_list = []
    files_names = []
    search_string = "{}/{}_normal_{}_*_level_{}*{}*test_acc_*.pth".format(args.folder, args.model, args.dataset,
                                                                          args.RF_level, args.name)
    things = list(glob.glob(search_string))

    # if len(things) < 2:
    #     search_string = "{}/{}_normal_{}_*_level_{}.pth".format(args.folder, args.model, args.dataset, args.RF_level)

    print("Glob text:{}".format(
        "{}/{}_normal_{}_*_level_{}*{}*test_acc_*.pth".format(args.folder, args.model, args.dataset, args.RF_level,
                                                              args.name)))
    print(things)

    for i, name in enumerate(
            glob.glob(search_string)):

        print(name)

        print("Device: {}".format(device))

        state_dict_raw = torch.load(name, map_location=device)

        dense_accuracy_list.append(state_dict_raw["acc"])

        print("Dense accuracy:{}".format(state_dict_raw["acc"]))

        net.load_state_dict(state_dict_raw["net"])
        prune_function(net, cfg)
        remove_reparametrization(net, exclude_layer_list=cfg.exclude_layers)

        t0 = time.time()
        if args.ffcv:
            pruned_accuracy = test_ffcv(net, testloader=testloader, verbose=0)
        else:
            pruned_accuracy = test(net, use_cuda=use_cuda, testloader=testloader, verbose=0)

        t1 = time.time()

        print("Pruned accuracy:{}".format(pruned_accuracy))

        print("Time for inference: {}".format(t1 - t0))

        pruned_accuracy_list.append(pruned_accuracy)
        weight_names, weights = zip(*get_layer_dict(net))

        zero_number = lambda w: (torch.count_nonzero(w == 0) / w.nelement()).cpu().numpy()

        pruning_rates_per_layer = list(map(zero_number, weights))

        seed_from_file1 = re.findall("_[0-9]_", name)

        print(seed_from_file1)

        seed_from_file2 = re.findall("_[0-9]_[0-9]_", name)

        print(seed_from_file2)

        seed_from_file3 = re.findall("\.[0-9]_", name)

        print(seed_from_file3)

        if seed_from_file3:

            seed_from_file = seed_from_file3[0].replace(".", "_")

        elif seed_from_file2:

            seed_from_file = seed_from_file2[0].split("_")[2]

        elif seed_from_file1:

            seed_from_file = seed_from_file1[0].replace("_", "")
        else:
            seed_from_file = i

        df2 = pd.DataFrame({"layer_names": weight_names, "pr": pruning_rates_per_layer})
        print("Seed from file {}".format(seed_from_file1))
        df2.to_csv(
            "{}_level_{}_seed_{}_{}_{}_pruning_rates_global_pr_{}.csv".format(args.model, args.RF_level, seed_from_file,
                                                                              args.dataset, args.name,
                                                                              args.pruning_rate),
            index=False)

        print("Done")
        file_name = os.path.basename(name)
        print(file_name)
        files_names.append(file_name)

    df = pd.DataFrame({"Name": files_names,
                       "Dense Accuracy": dense_accuracy_list,
                       "Pruned Accuracy": pruned_accuracy_list,
                       })
    df.to_csv("RF_{}_{}_{}_{}_one_shot_summary.csv".format(args.model, args.RF_level, args.dataset, args.pruning_rate),
              index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='One shot pruning statistics')

    parser.add_argument('--experiment', default=1, type=int, help='Experiment to perform')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--type', default="normal", type=str, help='Type of implementation [normal,official]')
    parser.add_argument('--RF_level', default="4", type=str, help='Receptive field level')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--dataset', default="cifar10", type=str, help='Dataset to use [cifar10,tiny_imagenet616gg]')
    parser.add_argument('--model', default="resnet18", type=str, help='Architecture of model [resnet18,resnet50]')
    parser.add_argument('--folder', default="/nobackup/sclaam/checkpoints", type=str,
                        help='Location where saved models are')
    parser.add_argument('--name', default="", type=str, help='Name of the file', required=False)
    parser.add_argument('--solution', default="", type=str, help='Solution to use')
    parser.add_argument('--pruning_rate', default=0.9, type=float, help='Pruning rate')
    parser.add_argument('--epochs', default=200, type=int, help='Epochs to train')
    parser.add_argument('--width', default=1, type=int, help='Width of the model')
    parser.add_argument('--data_folder', default="/nobackup/sclaam/data", type=str,
                        help='Location to save the models', required=True)
    parser.add_argument('--ffcv', action='store_true', help='Use FFCV loaders')

    parser.add_argument('--ffcv_train',
                        default="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/train_360_0.5_90.ffcv",
                        type=str, help='FFCV train dataset')

    parser.add_argument('--ffcv_val',
                        default="/jmain02/home/J2AD014/mtc03/lla98-mtc03/small_imagenet_ffcv/val_360_0.5_90.ffcv",
                        type=str, help='FFCV val dataset')

    args = parser.parse_args()

    try:

        args.RF_level = int(args.RF_level)

    except Exception as e:

        pass

    if args.experiment == 1:
        print("Experiment 1")
        print(args)
        main(args)
        # pruning_fine_tuning_experiment(args)
    # if args.experiment == 2:
    #     print("Experiment 2")
    #     print(args)
    #     # main(args)
    #     pruning_fine_tuning_experiment(args)
    # if args.experiment == 3:
    #     # pruning_fine_tuning_experiment(args)
    #     fine_tune_summary(args)
    # if args.experiment == 4:
    #     n_shallow_layer_experiment(args)
    #     # main(args)
    # # gradient_flow_calculation(args)
    # # save_pruned_representations()
    # # similarity_comparisons()
