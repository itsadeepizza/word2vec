import torch
import torch.nn as nn
from dataset import dataloader
from dataset import dataset
from dataset import vocab
from model import Model

if torch.cuda.is_available(): device = torch.device("cuda") 

use_tensorboard = True
model = Model(device=device,len_voc=len(vocab))
opt = torch.optim.SGD(model.parameters(), lr = 0.01)
criterion = torch.nn.L1Loss()

if use_tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    from torchvision.transforms import ToTensor
    import matplotlib.pyplot as plt
    import PIL.Image
    import io
    import matplotlib 

    writer = SummaryWriter()
    writer.add_graph(model,next(iter(dataloader))[0])
    matplotlib.use('Agg')

    def gen_plot(embedding,word1,word2,word3,step):
        with torch.no_grad():
            
            #esperimenti con tensorboard
            #mat = torch.stack((embedding(vocab[word1]),embedding(vocab[word2]),embedding(vocab[word3])))
            #writer.add_embedding(mat,global_step=step)
            #writer.add_mesh("embedding", mat[None,:],global_step=step)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            (v1,v2,v3) = (embedding(vocab[word1]).tolist(), \
                       embedding(vocab[word2]).tolist(), \
                       embedding(vocab[word3]).tolist())
            #only 3 dim of embedding
            #v1 = [v1[1],v1[3],v1[4]]
            #v2 = [v2[1],v2[3],v2[4]]
            #v3 = [v3[1],v3[3],v3[4]]

            v1.insert(0,0)
            v1.insert(0,0)
            v1.insert(0,0)
            v2.insert(0,0)
            v2.insert(0,0)
            v2.insert(0,0)
            v3.insert(0,0)
            v3.insert(0,0)
            v3.insert(0,0)
            X_0,Y_0,Z_0, X,Y,Z = zip(v1,v2,v3)
            ax.quiver(X_0[0],Y_0[0],Z_0[0],X[0],Y[0],Z[0], normalize = True,color="blue")
            ax.quiver(X_0[1],Y_0[1],Z_0[1],X[1],Y[1],Z[1], normalize = True,color="red")
            ax.quiver(X_0[2],Y_0[2],Z_0[2],X[2],Y[2],Z[2], normalize = True,color="green")
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            plt.title("is(blue) - was(red) - a(green)")
            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            plt.close('all')
            return buf

with torch.no_grad():
    print("before training")
    emb_is = model.embedding(vocab["is"])
    emb_was = model.embedding(vocab["was"])
    print(emb_is)
    print(emb_was)
    print(torch.dot(emb_is,emb_was))

# training loop
tot_iters = 0
for epoch in range(50):
    print("epoch started: ",epoch)
    data = iter(dataloader)
    for i,(words,label) in enumerate(data):
        out = model(words)
        loss = criterion(out, label.to(device))
        if i % 1000 == 0:
            print(f"epoch {epoch} iteration {i} loss:",loss.item())
            if use_tensorboard:
                index = epoch*tot_iters + i
                plot_buf = gen_plot(model.embedding,"is","was","airport",index)
                image = PIL.Image.open(plot_buf)
                image = ToTensor()(image).unsqueeze(0)[0]
                writer.add_image('plot embedding', image, index)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if i>tot_iters:
            tot_iters = i
    # TODO save model / PCA prima di plottare

if use_tensorboard:
    writer.close()

with torch.no_grad():
    print("after training")
    emb_is = model.embedding(vocab["is"])
    emb_was = model.embedding(vocab["was"])
    print(emb_is)
    print(emb_was)
    print(torch.dot(emb_is,emb_was))
