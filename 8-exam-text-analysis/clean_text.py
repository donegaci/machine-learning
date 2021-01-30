import itertools  
import re
import pickle


with open("english.pickle", "rb") as f:
    X,y,z = pickle.load(f)

for i, (text, polarity, early_access) in enumerate(zip(X,y,z)):
    cleaned = re.sub("[^a-zA-Z ]+", "", text)
    if cleaned=="" or cleaned.isspace():
        X.pop(i)
        y.pop(i)
        z.pop(i)
    else:
        X[i] = cleaned

print(len(X))
print(len(y))
print(len(z))

with open("cleaned.pickle", "wb") as f:
    pickle.dump([X, y, z], f)

