import pandas as pd

gramSize = 4
http = "http://"
httpFreq = 0
https = "https://"
httpsFreq = 0
freqs = {}

def AddToDict(key):
    global freqs
    freqs.update({key: 1})
    return 1

def IsInDict(key):
    global freqs
    return key in freqs

def IncDictEntry(key):
    global freqs
    freqs[key] += 1

def RemovePrefix(prefix, string):
    strTmp = string[0 : len(prefix)]
    idx = strTmp.find(prefix)
    if idx != -1:
        string = string[idx + len(string) :]
    return idx != -1

def CreateDict(url):
    global httpFreq
    global httpsFreq

    if RemovePrefix(https, url) == False:
        if RemovePrefix(http, url) == True:
            httpFreq += 1
    else:
        httpsFreq += 1
    urlSize = len(url)

    if urlSize < gramSize:
        print('No {0}-gram extracted from {1}'.format(gramSize, url))
        return

    for i in range(urlSize - gramSize):
        key = url[i : i + gramSize]
        if (IsInDict(key)):
            IncDictEntry(key)
        else:
            AddToDict(key)

    return 1

df = pd.read_csv("./dataset/bb.csv/brasilia.csv",
                sep = ',',
                usecols = ['url'],
                encoding = 'utf-8')

df.apply(lambda line : line.map(CreateDict), axis = 1)

cont = 0
cols = []
rows = []
for k, v in sorted(freqs.items(), key = lambda it : it[1], reverse = True):
    if cont > 256:
        break
    cols.append(k)
    rows.append(v)
    cont += 1

freqsDf = pd.DataFrame(data = [rows], columns = cols)
freqsDf.to_csv("freqs.csv", index = False)

print('freqs dictionary size [{}]'.format(len(freqs)))
print('https freq [{}]'.format(httpsFreq))
print('http freq [{}]'.format(httpFreq))