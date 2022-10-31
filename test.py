import torch
import matplotlib.pyplot as plt
from dataset import vocab
from model import Model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_file = "model.pth"
model = Model(device=device, len_voc=len(vocab))
print(f"loading model from file {model_file}")
model.load_state_dict(torch.load(model_file))

def embed(word):
    with torch.no_grad():
        return model.embedding(vocab[word]).cpu()

def find_best_by_dist(embedded_word):
    min_dist = None
    best = None
    for element in vocab:
        delta = embed(element)-embedded_word
        delta_mod = torch.linalg.vector_norm(delta,ord=2)
        if min_dist == None:
            best = element
            min_dist = delta_mod
        elif min_dist and delta_mod<min_dist:
            min_dist = delta_mod
            best = element
    return best

# ?
#def dotprod(word1, word2):
#    return model(torch.cat((vocab[word1].unsqueeze(0), vocab[word2].unsqueeze(0)), dim=0).unsqueeze(0))

def show(words):
    fig, ax = plt.subplots(len(words))
    for i, word in enumerate(words):
        ax[i].set_title(word)
        ax[i].imshow(embed(word).unsqueeze(0))
        ax[i].set_axis_off()

#def tovec(word):
#    v = model.embedding(vocab[word]).detach()
#    return v / torch.norm(v)

show(["is", "was", "airport", "store"])
show(["king", "queen", "woman", "store"])
show(["man", "woman", "women", "actor", "sleep"])
show(["is", "was", "do", "go", "be", "a", "the"])
plt.show() # show all figures and wait
print("king - man + woman =",find_best_by_dist(embed("king")-embed("man")+embed("woman")))
