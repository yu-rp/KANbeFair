import logging, os, sys, gc, time, re
from datetime import datetime
import torch, random, numpy, torchtext
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset, Subset
import matplotlib.pyplot as plt

from fvcore.nn import FlopCountAnalysis, parameter_count

from models.mlp import *
from models.kanbefair import *
from models.bspline_mlp import *
from models.utils import *

from data.titanic import TitanicDataset
from data.uciml import from_uciml_to_dataset
from data.special import get_scipyfunction_dataset, get_special_dataset_1d
from data.pca_dataset import get_pca_dataset
from data.text import create_text_loader, get_IMDb_dataset
from data.audio import SubsetSC, get_SC_loader, get_US_dataset

def get_timestamp():
    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_time

def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    print('Log directory: ', log_dir)
    return logger, formatter

def get_loader(args, shuffle = True, use_cuda = True):
    train_kwargs = {'batch_size': args.batch_size, 'num_workers': 4}
    test_kwargs = {'batch_size': args.test_batch_size, 'num_workers': 4}

    if shuffle:
        train_kwargs.update({'shuffle': True})
        test_kwargs.update({'shuffle': False})
    else:
        train_kwargs.update({'shuffle': False})
        test_kwargs.update({'shuffle': False})

    if args.dataset == "MNIST":
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.MNIST('../dataset', train=True, download=True,
                        transform=transform)
        test_dataset = datasets.MNIST('../dataset', train=False, download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 10
        input_size = 1 * 28 * 28
    elif "MNIST_PCA" in args.dataset:
        n_components = int(args.dataset.rsplit("_",1)[1])
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.MNIST('../dataset', train=True, download=True,
                        transform=transform)
        test_dataset = datasets.MNIST('../dataset', train=False, download=True,
                        transform=transform)

        train_dataset, test_dataset = get_pca_dataset(train_dataset, test_dataset, n_components)
        
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 10
        input_size = n_components
        
    elif args.dataset == "Titanic":
        train_dataset, test_dataset = TitanicDataset()
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 2
        input_size = 9
    elif args.dataset == "EMNIST-Letters":
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1724,), (0.3311,)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.EMNIST('../dataset', split = "letters", train=True, download=True,
                        transform=transform)
        test_dataset = datasets.EMNIST('../dataset', split = "letters", train=False, download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 37
        input_size = 1 * 28 * 28
    elif args.dataset == "EMNIST-Balanced":
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1753,), (0.3334,)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.EMNIST('../dataset', split = "balanced",  train=True, download=True,
                        transform=transform)
        test_dataset = datasets.EMNIST('../dataset', split = "balanced",  train=False, download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 47
        input_size = 1 * 28 * 28
    elif args.dataset == "FMNIST":
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.FashionMNIST('../dataset', train=True, download=True,
                        transform=transform)
        test_dataset = datasets.FashionMNIST('../dataset', train=False, download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 10
        input_size = 1 * 28 * 28
    elif args.dataset == "KMNIST":
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1918,), (0.3483,)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.KMNIST('../dataset', train=True, download=True,
                        transform=transform)
        test_dataset = datasets.KMNIST('../dataset', train=False, download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 10
        input_size = 1 * 28 * 28
    elif args.dataset == "Cifar10":
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Resize(28),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.CIFAR10('../dataset', train=True, download=True,
                        transform=transform)
        test_dataset = datasets.CIFAR10('../dataset', train=False, download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 10
        input_size = 3 * 28 * 28
    elif args.dataset == "Cifar100":
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Resize(28),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.CIFAR100('../dataset', train=True, download=True,
                        transform=transform)
        test_dataset = datasets.CIFAR100('../dataset', train=False, download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 100
        input_size = 3 * 28 * 28
    elif args.dataset == "DTD":
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize((0.5283, 0.4738, 0.4231), (0.2689, 0.2596, 0.2669)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.DTD('../dataset', split ="train", download=True,
                        transform=transform)
        test_dataset = datasets.DTD('../dataset', split ="test", download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 47
        input_size = 3 * 224 * 224
    elif args.dataset == "Pet":
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize((0.4845, 0.4529, 0.3958), (0.2686, 0.2645, 0.2735)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.OxfordIIITPet('../dataset', split ="trainval", download=True,
                        transform=transform)
        test_dataset = datasets.OxfordIIITPet('../dataset', split ="test", download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 37
        input_size = 3 * 224 * 224
    elif args.dataset == "SVHN":
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.4381, 0.4442, 0.4734), (0.1983, 0.2013, 0.1972)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.SVHN('../dataset', split ="train", download=True,
                        transform=transform)
        test_dataset = datasets.SVHN('../dataset', split ="test", download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 10
        input_size = 3 * 28 * 28
    elif args.dataset == "TinyImageNet":
        transform=transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2764, 0.2689, 0.2816)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.ImageFolder('../dataset/tiny-imagenet-200/train', transform=transform)
        test_dataset = datasets.ImageFolder('../dataset/tiny-imagenet-200/tidy_val', transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 200
        input_size = 3 * 28 * 28
    elif args.dataset == "Abalone":
        train_dataset, test_dataset = from_uciml_to_dataset(1)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 28
        input_size = 8
    elif args.dataset == "Income":
        train_dataset, test_dataset = from_uciml_to_dataset(2)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 2
        input_size = 14
    elif args.dataset == "Mushroom":
        train_dataset, test_dataset = from_uciml_to_dataset(73)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 2
        input_size = 22
    elif args.dataset == "Wine":
        train_dataset, test_dataset = from_uciml_to_dataset(186)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 7
        input_size = 11
    elif args.dataset == "Bank":
        train_dataset, test_dataset = from_uciml_to_dataset(222)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 2
        input_size = 16
    elif args.dataset == "Rice":
        train_dataset, test_dataset = from_uciml_to_dataset(545)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 2
        input_size = 7
    elif args.dataset == "Bean":
        train_dataset, test_dataset = from_uciml_to_dataset(602)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 7
        input_size = 16
    elif args.dataset == "Student":
        train_dataset, test_dataset = from_uciml_to_dataset(697)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 3
        input_size = 36
    elif args.dataset == "Spam":
        train_dataset, test_dataset = from_uciml_to_dataset(94)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 2
        input_size = 57
    elif args.dataset == "Card":
        train_dataset, test_dataset = from_uciml_to_dataset(350)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 2
        input_size = 23
    elif args.dataset == "Telescope":
        train_dataset, test_dataset = from_uciml_to_dataset(159)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 2
        input_size = 10
    elif args.dataset == "Dota":
        train_dataset, test_dataset = from_uciml_to_dataset(367)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 2
        input_size = 115
    elif args.dataset == "Darwin":
        train_dataset, test_dataset = from_uciml_to_dataset(732)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 2
        input_size = 451
    elif args.dataset == "Toxicity":
        train_dataset, test_dataset = from_uciml_to_dataset(728)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 2
        input_size = 1203
    elif args.dataset in [
            "Special_ellipj",
            "Special_ellipkinc",
            "Special_ellipeinc",
            "Special_jv",
            "Special_yv",
            "Special_kv",
            "Special_iv",
            "Special_lpmv0",
            "Special_lpmv1",
            "Special_lpmv2",
            "Special_sphharm01",
            "Special_sphharm11",
            "Special_sphharm02",
            "Special_sphharm12",
            "Special_sphharm22",
            ]:
        train_dataset, test_dataset = get_scipyfunction_dataset(args)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 1
        input_size = 2
    elif args.dataset in ["Special_1d_poisson","Special_1d_gelu"]:
        train_dataset, test_dataset = get_special_dataset_1d(args)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 1
        input_size = 1
    elif args.dataset == "AG_NEWS":
        train_dataset = torchtext.datasets.AG_NEWS(root = "../dataset", split='train')
        train_dataset = [(int(item[0])-1, item[1]) for item in train_dataset]

        test_dataset = torchtext.datasets.AG_NEWS(root = "../dataset", split='test')
        test_dataset = [(int(item[0])-1, item[1]) for item in test_dataset]

        train_loader, vocab = create_text_loader(train_dataset, None, train_kwargs)
        test_loader, _ = create_text_loader(test_dataset, vocab, test_kwargs)
        num_classes = 4
        input_size = len(vocab) # for text classification, the input size is the number of vocabularies
    elif args.dataset == "CoLA":
        train_dataset = torchtext.datasets.CoLA(root = "../dataset", split='train')
        train_dataset = [(int(item[1]), item[2]) for item in train_dataset]

        test_dataset = torchtext.datasets.CoLA(root = "../dataset", split='test')
        test_dataset = [(int(item[1]), item[2]) for item in test_dataset]

        train_loader, vocab = create_text_loader(train_dataset, None, train_kwargs)
        test_loader, _ = create_text_loader(test_dataset, vocab, test_kwargs)
        num_classes = 2
        input_size = len(vocab)
    elif args.dataset == "IMDb":
        train_dataset, test_dataset = get_IMDb_dataset()

        train_loader, vocab = create_text_loader(train_dataset, None, train_kwargs)
        test_loader, _ = create_text_loader(test_dataset, vocab, test_kwargs)
        num_classes = 2
        input_size = len(vocab)
    elif args.dataset == "SpeechCommand":
        ## Without Cache
        # train_dataset = SubsetSC("training")
        # test_dataset = SubsetSC("testing")
        # train_loader = get_SC_loader(train_dataset, train_kwargs)
        # test_loader = get_SC_loader(test_dataset, test_kwargs)

        ## With Cache, use the audio.py to generate the cache
        train_dataset, test_dataset = torch.load("../dataset/SpeechCommands/train_sc.pt"),torch.load("../dataset/SpeechCommands/test_sc.pt")

        train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        num_classes = 35
        input_size = 1000
    elif args.dataset == "UrbanSound8K": # Need to be downloaded manually
        ## Without Cache
        # train_dataset, test_dataset = get_US_dataset()

        ## With Cache, use the audio.py to generate the cache
        train_dataset, test_dataset = torch.load("../dataset/UrbanSound8K/train_us.pt"),torch.load("../dataset/UrbanSound8K/test_us.pt")

        train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 10
        input_size = 1000
    else:
        raise NotImplementedError

    return train_loader, test_loader, num_classes, input_size


def get_continual_loader(args, shuffle = True, use_cuda = True):
    train_kwargs = {'batch_size': args.batch_size, 'num_workers': 4}
    test_kwargs = {'batch_size': args.test_batch_size, 'num_workers': 4}

    if shuffle:
        train_kwargs.update({'shuffle': True})
        test_kwargs.update({'shuffle': False})
    else:
        train_kwargs.update({'shuffle': False})
        test_kwargs.update({'shuffle': False})

    if args.dataset == "Class_MNIST":
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.MNIST('../dataset', train=True, download=True,
                        transform=transform)
        test_dataset = datasets.MNIST('../dataset', train=False, download=True,
                        transform=transform)
        train_labels = torch.Tensor(train_dataset.targets)
        test_labels = torch.Tensor(test_dataset.targets)
        train_index_dict = {}
        for i in range(10):
            train_index_dict[i] = (train_labels == i).nonzero().squeeze().tolist()
        test_index_dict = {}
        for i in range(10):
            test_index_dict[i] = (test_labels == i).nonzero().squeeze().tolist()
        train_datasets = [
            Subset(train_dataset, train_index_dict[0]+train_index_dict[1]+train_index_dict[2]),
            Subset(train_dataset, train_index_dict[3]+train_index_dict[4]+train_index_dict[5]),
            Subset(train_dataset, train_index_dict[6]+train_index_dict[7]+train_index_dict[8]),
        ]
        test_datasets = [
            Subset(test_dataset, test_index_dict[0]+test_index_dict[1]+test_index_dict[2]),
            Subset(test_dataset, test_index_dict[3]+test_index_dict[4]+test_index_dict[5]),
            Subset(test_dataset, test_index_dict[6]+test_index_dict[7]+test_index_dict[8]),
        ]
        train_loaders = [
            torch.utils.data.DataLoader(dataset,**train_kwargs) for dataset in train_datasets]
        test_loaders = [
            torch.utils.data.DataLoader(dataset, **test_kwargs) for dataset in test_datasets]
        num_classes = 9
        input_size = 1 * 28 * 28
    else:
        raise NotImplementedError

    return train_loaders, test_loaders, num_classes, input_size

def get_model(args):
    if args.model == "MLP":
        model = MLP(args)
    elif args.model == "KAN":
        model = KANbeFair(args)
    elif args.model == "MLP_Text":
        model = MLP_Text(args)
    elif args.model == "KAN_Text":
        model = KANbeFair_Text(args)
    elif args.model == "BSpline_MLP":
        model = BSpline_MLP(args)
    elif args.model == "BSpline_First_MLP":
        model = BSpline_First_MLP(args)
    else:
        raise NotImplementedError
    return model

def randomness_control(seed):
    print("seed",seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_matrix(matrix, path):
    fig, ax = plt.subplots()
    cax = ax.imshow(matrix, cmap='inferno')
    fig.colorbar(cax)
    fig.savefig(path)

def get_filename(path):
    base_name = os.path.basename(path)  # filename.extension
    file_name_without_extension = os.path.splitext(base_name)[0]
    return file_name_without_extension

def measure_time_memory(f):
    def wrapped(*args, **kwargs):
        if torch.cuda.is_available():
            start_memory = torch.cuda.memory_allocated()
            torch.cuda.reset_max_memory_allocated()
        else:
            start_memory = 0

        start_time = time.time()

        result = f(*args, **kwargs)

        end_time = time.time()

        if torch.cuda.is_available():
            end_memory = torch.cuda.max_memory_allocated()
        else:
            end_memory = 0

        print(f"Function {f.__name__} executed in {end_time - start_time:.4f} seconds.")
        print(f"Memory usage increased by {(end_memory - start_memory) / (1024 ** 2):.2f} MB to {(end_memory) / (1024 ** 2):.2f} MB.")
        
        return result
    return wrapped

def classwise_validation(logits, label, targets, args):
    accuracies = []
    for target in targets:
        accuracies.append(accuracy = get_accuracy(logits, label, target))
    return accuracies

def get_accuracy(probability, label, target = None):
    prediction = probability.max(dim = 1)[1]
    if target is None:
        return ((prediction == label).sum() / label.numel()).item()
    else:
        mask = label == target
        return ((prediction[mask]== label[mask]).sum() / label[mask].numel()).item()

def get_activation(args):
    if args.activation_name == 'relu':
        return nn.ReLU
    elif args.activation_name == 'square_relu':
        return Square_ReLU
    elif args.activation_name == 'sigmoid':
        return nn.Sigmoid
    elif args.activation_name == 'tanh':
        return nn.Tanh
    elif args.activation_name == 'softmax':
        return nn.Softmax(dim=1)
    elif args.activation_name == 'silu':
        return nn.SiLU
    elif args.activation_name == 'gelu':
        return nn.GELU
    elif args.activation_name == 'glu':
        return nn.GLU
    elif args.activation_name == 'polynomial2':
        return Polynomial2
    elif args.activation_name == 'polynomial3':
        return Polynomial3
    elif args.activation_name == 'polynomial5':
        return Polynomial5
    else:
        raise ValueError(f'Unknown activation function: {args.activation_name}')

def get_shortcut_function(args):
    if args.kan_shortcut_name == 'silu':
        return nn.SiLU()
    elif args.kan_shortcut_name == 'identity':
        return nn.Identity()
    elif args.kan_shortcut_name == 'zero':

        class Zero(nn.Module):
            def __init__(self):
                super(Zero, self).__init__()
            def forward(self, x):
                return x * 0

        return Zero()
    else:
        raise ValueError(f'Unknown kan shortcut function: {args.kan_shortcut_name}')
    
def get_model_complexity(model, logger, args, method = "coustomized"):

    # using fvcore
    if method == "fvcore":
        parameter_dict = parameter_count(model)
        num_parameters = parameter_dict[""]

        flops_dict = FlopCountAnalysis(model, torch.randn(2, args.input_size))
        flops = flops_dict.total()
    elif method == "coustomized":
        num_parameters = model.total_parameters()
        flops = model.total_flops()
    else:
        raise NotImplementedError

    if logger is not None:
        logger.info(f"Number of parameters: {num_parameters:,}; Number of FLOPs: {flops:,}")

    return num_parameters, flops

def write_results(args, subfix = "", **kwargs):
    result_base = "../results"
    result_file = f"results{subfix}.csv"

    dataset, model, general_parameters, specific_parameter = args.exp_id.split("/")[2:]
    general_parameters = general_parameters.split("__")
    specific_parameter = specific_parameter.split("__")

    result_file_path = os.path.join(result_base, result_file)
    
    s = [get_timestamp(), dataset, model] + general_parameters + specific_parameter + [str(kwargs[key]) for key in kwargs]
    s = ",".join(s) + "\n"

    with open(result_file_path, "a") as f:
        f.write(s)

def todevice(obj, device):
    if isinstance(obj, (list,tuple)):
        obj = [o.to(device) for o in obj]
    elif isinstance(obj, torch.Tensor):
        obj = obj.to(device)
    else:
        raise NotImplementedError
    return obj