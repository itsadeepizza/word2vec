import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# utils
def get_vocab(vocfile):
    print("reading dataset")
    try:
        voc = pd.read_csv(vocfile, sep=";", header=None)
    except FileNotFoundError:
        print("file not found: creating dataset and vocab")
        from makeds import run
        run()
        voc = pd.read_csv(vocfile, sep=";", header=None)
    result = {}
    vocab = voc.get(0)
    for i, element in enumerate(vocab):
        result[element] = one_hot(i, len(vocab))
    return result

def one_hot(index,size):
    res = []
    for _ in range(size):
        res.append(0)
    res[index] = 1
    return torch.Tensor(res)

class CustomDataset(Dataset):
    def __init__(self, file):
        try:
            print("reading dataset")
            self.f_dataset = pd.read_csv("dataset.csv", sep=";", skipinitialspace=True)
            self.num_samples = len(self.f_dataset)
        except FileNotFoundError:
            from makeds import run
            print("file not found: creating dataset & vocab")
            run()
        except:
            print("cant open file :C")
            quit()

    def __getitem__(self, index):
        df_row = self.f_dataset.iloc[index].to_list()
        result = (torch.stack((vocab[df_row[0]], vocab[df_row[1]])), torch.Tensor([df_row[2]]))
        return result
        
    def __len__(self):
        return self.num_samples

vocab = get_vocab("vocab.csv")
dataset = CustomDataset("dataset.csv")
dataloader = DataLoader(dataset=dataset, batch_size=200, shuffle=True)

if __name__=="__main__":
    print(len(vocab))
