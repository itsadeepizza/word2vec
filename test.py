import torch
from dataset import vocab
from model import Model
from dataset import vocab


if torch.cuda.is_available(): device = torch.device("cuda")
device = torch.device("cpu")

use_tensorboard = True
use_existing_model = True
save_freq = 10000
model_file = "model.pth"
model = Model(device=device, len_voc=len(vocab))
if use_existing_model:
    print(f"loading model from file {model_file}")
    model.load_state_dict(torch.load(model_file))

def embed(word):
    return model.embedding(vocab["word"]).to('cpu').detach()

def dotprod(word1, word2):
    return model(torch.cat((vocab[word1].unsqueeze(0), vocab[word2].unsqueeze(0)), dim=0).unsqueeze(0))

def show(words):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(len(words))
    for i, word in enumerate(words):
        ax[i].set_title(word)
        ax[i].imshow(model.embedding(vocab[word]).detach().unsqueeze(0))
        ax[i].set_axis_off()
    fig.show()

def tovec(word):
    v = model.embedding(vocab[word]).detach()
    return v / torch.norm(v)

show(["is", "was", "airport", "store"])
show(["king", "queen", "woman", "store"])
show(["man", "woman", "women", "actor", "sleep"])
show(["is", "was", "do", "go", "be", "a", "the"])