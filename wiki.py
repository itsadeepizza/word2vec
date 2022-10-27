import requests, json
from bs4 import BeautifulSoup

def get_article(i, verbose = False):
    i = int(i)
    if verbose: print("[wikipedia] getting source - id "+str(i))
    link = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&pageids="+str(i)+"&inprop=url&format=json"
    text=requests.get(link).text
    jsonobj = json.loads(text)
    title = jsonobj['query']['pages'][str(i)]['title']
    if verbose: print("[wikipedia] link: ",link, " title: ", title)
    html = jsonobj['query']['pages'][str(i)]['extract']
    parsed = BeautifulSoup(html, "html.parser")
    par = parsed.find_all("p")
    res = ""
    for element in par:
        if element.text.isascii():
            res += element.text
    return res
    """
    if verbose: print("[wikipedia] putting into file - id "+str(i))
    with open("wikipedia/"+str(i)+"--id.json","w", encoding="utf-8") as f:
        f.writelines(res)
        print("[wikipedia] archived - id "+str(i))
    """
def gen_article():
    for i in range(10000,1_000_000):
        # i<1000 = api error, <10000 more weird data
        try:
            art = get_article(i,verbose = False)
            if len(art.replace("\n","").replace(" ","")) == 0:  # dont return only whitespaces + \n
                continue
            yield art
        except KeyError:
            continue
         

article = gen_article()

if __name__=="__main__":
    for i in range(10): 
        print(next(article))
        print("-" * 80)

  
