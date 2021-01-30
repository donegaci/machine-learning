import json_lines
import pickle
from google_trans_new import google_translator
import re
import time

def load_and_translate(path):
    X = []
    y = []
    z = []
    translator = google_translator() 
    start = time.time()
    with open(path, "rb") as f:
        for item in json_lines.reader(f):
            english = translator.translate(item['text'])
            english = re.sub("[^0-9a-zA-Z ]+", "", english)
            if not(english=="" and english.isspace):
                X.append(english)
                y.append(item['voted_up'])
                z.append(item['early_access'])
            else:
                print(english)

    
    print(len(X))
    with open("translated.pickle", "wb") as f:
        pickle.dump([X, y, z], f)
        
    print (time.time() - start)
    return


load_and_translate("data.json")

