from scipy import special
import torch, random, numpy
from torch.utils.data import Dataset, TensorDataset, Subset

def get_scipyfunction_dataset(args):
    train_x = numpy.random.rand(1000, 2)
    test_x = numpy.random.rand(1000, 2)

    if  args.dataset == "Special_ellipj":
        train_y = special.ellipj(train_x[:,0], train_x[:,1])
        train_y = numpy.stack(train_y, axis=1)
        test_y = special.ellipj(test_x[:,0], test_x[:,1])
        test_y = numpy.stack(test_y, axis=1)
        raise ValueError("Unsupported dataset")
    elif args.dataset == "Special_ellipkinc":
        train_y = special.ellipkinc(train_x[:,0], train_x[:,1])
        train_y = numpy.expand_dims(train_y, axis=1)
        test_y = special.ellipkinc(test_x[:,0], test_x[:,1])
        test_y = numpy.expand_dims(test_y, axis=1)
    elif args.dataset == "Special_ellipeinc":
        train_y = special.ellipeinc(train_x[:,0], train_x[:,1])
        train_y = numpy.expand_dims(train_y, axis=1)
        test_y = special.ellipeinc(test_x[:,0], test_x[:,1])
        test_y = numpy.expand_dims(test_y, axis=1)
    elif args.dataset == "Special_jv":
        train_y = special.jvp(train_x[:,0], train_x[:,1])
        train_y = numpy.expand_dims(train_y, axis=1)
        test_y = special.jvp(test_x[:,0], test_x[:,1])
        test_y = numpy.expand_dims(test_y, axis=1)
    elif args.dataset == "Special_yv":
        train_y = special.yvp(train_x[:,0], train_x[:,1])
        train_y = numpy.expand_dims(train_y, axis=1)
        test_y = special.yvp(test_x[:,0], test_x[:,1])
        test_y = numpy.expand_dims(test_y, axis=1)
    elif args.dataset == "Special_kv":
        train_y = special.kvp(train_x[:,0], train_x[:,1])
        train_y = numpy.expand_dims(train_y, axis=1)
        test_y = special.kvp(test_x[:,0], test_x[:,1])
        test_y = numpy.expand_dims(test_y, axis=1)
    elif args.dataset == "Special_iv":
        train_y = special.ivp(train_x[:,0], train_x[:,1])
        train_y = numpy.expand_dims(train_y, axis=1)
        test_y = special.ivp(test_x[:,0], test_x[:,1])
        test_y = numpy.expand_dims(test_y, axis=1)
    elif args.dataset == "Special_lpmv0":
        train_y = special.lpmv(0, train_x[:,0], train_x[:,1])
        train_y = numpy.expand_dims(train_y, axis=1)
        test_y = special.lpmv(0, test_x[:,0], test_x[:,1])
        test_y = numpy.expand_dims(test_y, axis=1)
    elif args.dataset == "Special_lpmv1":
        train_y = special.lpmv(1, train_x[:,0], train_x[:,1])
        train_y = numpy.expand_dims(train_y, axis=1)
        test_y = special.lpmv(1, test_x[:,0], test_x[:,1])
        test_y = numpy.expand_dims(test_y, axis=1)
    elif args.dataset == "Special_lpmv2":
        train_y = special.lpmv(2, train_x[:,0], train_x[:,1])
        train_y = numpy.expand_dims(train_y, axis=1)
        test_y = special.lpmv(2, test_x[:,0], test_x[:,1])
        test_y = numpy.expand_dims(test_y, axis=1)
    elif args.dataset == "Special_sphharm01":
        train_y = special.sph_harm(0, 1, train_x[:,0], train_x[:,1])
        train_y = numpy.expand_dims(train_y.real, axis=1)
        test_y = special.sph_harm(0, 1, test_x[:,0], test_x[:,1])
        test_y = numpy.expand_dims(test_y.real, axis=1)
    elif args.dataset == "Special_sphharm02":
        train_y = special.sph_harm(0, 2, train_x[:,0], train_x[:,1])
        train_y = numpy.expand_dims(train_y.real, axis=1)
        test_y = special.sph_harm(0, 2, test_x[:,0], test_x[:,1])
        test_y = numpy.expand_dims(test_y.real, axis=1)
    else:
        raise ValueError("Unknown dataset")

    train_x = torch.from_numpy(train_x).float()
    test_x = torch.from_numpy(test_x).float()
    train_y = torch.from_numpy(train_y).float()
    test_y = torch.from_numpy(test_y).float()

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)

    return train_dataset, test_dataset

def get_special_dataset_1d(args):

    if  args.dataset == "Special_1d_poisson":
        train_x = torch.rand(1000, 1) * 8
        test_x = torch.rand(1000, 1) * 8
        def poisson(x):
            return 2**x / torch.exp(torch.lgamma(x + 1))
        train_y = poisson(train_x)
        test_y = poisson(test_x)
        train_x = train_x - 4
        test_x = test_x - 4
    elif args.dataset == "Special_1d_gelu":
        train_x = torch.rand(1000, 1) * 8 - 4
        test_x = torch.rand(1000, 1) * 8 - 4
        train_y = torch.nn.functional.gelu(train_x)
        test_y = torch.nn.functional.gelu(test_x)
    else:
        raise ValueError("Unknown dataset")

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)

    return train_dataset, test_dataset