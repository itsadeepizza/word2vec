from wiki import article
import random
import re
from tqdm import tqdm

window_size = 5
vocab_size = 15000
neg_sampl = 5 # continous bag of words with negative samples

words = []
vocab = {}

def make_csv(vocab): # very slow
    print("writing dataset")
    f = open("dataset.csv", "w")
    f.write("target" + "; "+ "word" + "; " + "is_close"+"\n")
    for element in tqdm(vocab,ascii=True, desc="dataset"):
        for close_word in vocab[element]:
            f.write(element + "; "+ close_word + "; " + "1"+"\n")
            neg_samples = []
            while len(neg_samples) < neg_sampl:
                neg_word = random.choice(list(vocab.keys()))
                if neg_word not in neg_samples and neg_word not in vocab[element]:
                    neg_samples.append(neg_word)
            for word in neg_samples: 
                f.write(element + "; "+ word + "; " + "0" + "\n")
    f.close()
def process_word(word):
    word = re.sub(r'[^\w\s]', '', word)
    return word.lower()

def run():
    while len(vocab) < vocab_size:
        words = []
        while len(words) < 100: 
            art = next(article)
            # TODO process text
            words = art.split()
        
        words = [process_word(word) for word in words]
        words = [word for word in words if word != ""]
        
        for i in range(0,len(words)-window_size):
            window = words[i:i+window_size]
            target = window[2]
            if target not in vocab.keys():
                vocab[target] = []
            for element in window:
                if element not in vocab[target] and element != target:
                    vocab[target].append(element)
                    if element not in vocab.keys():
                        vocab[element] = []
            if len(vocab) == vocab_size: break
    
        print("vocab size: ", len(vocab))
    # save all words in separate file
    print("writing vocab")
    f = open("vocab.csv", "w")
    for key in vocab.keys():
        f.write(key+"\n")
    f.close()
    make_csv(vocab)

if __name__=="__main__":
    run()
