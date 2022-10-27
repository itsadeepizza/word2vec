from wiki import article
import random

window_size = 5
vocab_size = 15000
neg_sampl = 5 # continous bag of words

words = []
vocab = {}

def make_csv(vocab):
    f = open("dataset.csv", "w")
    f.write("target" + "; "+ "word" + "; " + "is_close"+"\n")
    for element in vocab:
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
    word = word.replace("\"","")
    word = word.replace("\'","")
    word = word.replace('\n',"")
    word = word.replace("(","")
    word = word.replace(")","")
    word = word.replace("{","")
    word = word.replace("}","")
    word = word.replace("[","")
    word = word.replace("]","")
    word = word.replace(":","")
    word = word.replace(",","")
    word = word.replace(".","")
    word = word.replace(";","")
    word = word.replace("*","")
    word = word.replace("=","")
    word = word.lower()
    return word

while len(vocab) < vocab_size:
    breakpoint()
    words = []
    while len(words) < 100: 
        art = next(article)
        # TODO process text
        words = art.split()
    
    for i,word in enumerate(words):
        words[i] = process_word(word)

    # rifare in list comp
    #for i,word in enumerate(words): 
    #    if word == "":
    #        words.pop(i)
    words = [word for word in words if word != ""]
    
    for i in range(0,len(words)-window_size):
        window = words[i:i+window_size]
        target = window[2]
        if target not in vocab.keys() and target != "":
            vocab[target] = []
        for element in window:
            if target != "" and element != "" and element not in vocab[target] and element != target:
                vocab[target].append(element)
                if element not in vocab.keys():
                    vocab[element] = []
        if len(vocab) == vocab_size: break

    # save all words in separate file
    f = open("vocab.csv", "w")
    for key in vocab.keys():
        f.write(key+"\n")
    f.close()
    print("vocab size: ", len(vocab))

make_csv(vocab)

