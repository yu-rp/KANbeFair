import warnings
warnings.filterwarnings('ignore')

from ucimlrepo import fetch_ucirepo 
import torch, random, numpy
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, TensorDataset, Subset

def process_x(df,ucimlid):
    # category columns
    float_cols = [col for col in df.columns if df[col].dtype == 'float64']
    int_cols = [col for col in df.columns if df[col].dtype == 'int64']
    str_cols = [col for col in df.columns if df[col].dtype == 'object']
    other_cols = [col for col in df.columns if col not in float_cols + int_cols + str_cols]

    # drop columns with other types
    df = df.drop(other_cols, axis=1)

    # transform str_cols to int_cols
    for col in str_cols:
        df[col] = df[col].str.lower() 
        df[col] = df[col].str.strip('!"#%&\'()*,./:;?@[\\]^_`{|}~' + ' \n\r\t') 

    category_feature_map = [{} for cat in str_cols]

    def encode(encoder, x):
        len_encoder = len(encoder)
        try:
            id = encoder[x]
        except KeyError:
            id = len_encoder
        return id

    for i, cat in enumerate(str_cols):
        category_feature_map[i] = {l: id for id, l in enumerate(df.loc[:, cat].astype(str).unique())}
        df[cat] = df[cat].astype(str).apply(lambda x: encode(category_feature_map[i], x))
    
    # transform int_cols to float_cols
    for col in int_cols+str_cols:
        df[col] = df[col].astype(float)

    # fill missing values
    df[float_cols+int_cols+str_cols] = df[float_cols+int_cols+str_cols].fillna(0)

    # normalize float_cols
    scaler = StandardScaler()
    df[float_cols+int_cols+str_cols] = scaler.fit_transform(df[float_cols+int_cols+str_cols])
    
    # print(df.head())
    
    return df.values

def process_y(df,ucimlid):
    assert len(df.columns) == 1
    assert df.dtypes[0] in ['int64', 'object']

    # transform str_cols to int_cols
    if df.dtypes[0] == 'object':
        df[df.columns[0]] = df[df.columns[0]].str.lower() 
        df[df.columns[0]] = df[df.columns[0]].str.strip('!"#%&\'()*,./:;?@[\\]^_`{|}~' + ' \n\r\t') 

    def encode(encoder, x):
        len_encoder = len(encoder)
        try:
            id = encoder[x]
        except KeyError:
            id = len_encoder
        return id

    feature_map = {l: id for id, l in enumerate(df.loc[:, df.columns[0]].astype(str).unique())}
    df[df.columns[0]] = df[df.columns[0]].astype(str).apply(lambda x: encode(feature_map, x))

    # print(df.head())
    
    return df.values

def construct_dataset(x, y):
    
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)
    if len(y.shape) == 2:
        if y.shape[1] == 1:
            y = y.squeeze(1)
        elif y.shape[0] == 1:
            y = y.squeeze(0)
        else:
            raise ValueError('y should be 1D or 2D with shape[1]=1 or shape[0]=1.')
    elif len(y.shape) == 1:
        pass
    else:
        raise ValueError('y should be 1D or 2D with shape[1]=1 or shape[0]=1.')
    
    dataset = TensorDataset(x, y)
    
    return dataset

def split_dataset(dataset):

    train_idx, test_idx = torch.load('../dataset/uciml_split_idx.pt')

    train_idx = torch.tensor(train_idx)
    test_idx = torch.tensor(test_idx)

    train_idx = train_idx[train_idx < len(dataset)]
    test_idx = test_idx[test_idx < len(dataset)]

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    
    return train_dataset, test_dataset

def from_uciml_to_dataset(ucimlid):
    ucimldataset = fetch_ucirepo(id=ucimlid) 
  
    # data (as pandas dataframes) 
    x = ucimldataset.data.features 
    y = ucimldataset.data.targets 

    x = process_x(x,ucimlid)
    y = process_y(y,ucimlid)

    dataset = construct_dataset(x, y)
    train_dataset, test_dataset = split_dataset(dataset)
    return train_dataset, test_dataset