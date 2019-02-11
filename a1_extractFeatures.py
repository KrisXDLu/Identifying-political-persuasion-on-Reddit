import numpy as np
import sys
import argparse
import os
import json
import re
import csv
import string

def strToFloat_luxiaodi(data):
    # print(data)
    r = 0.0
    if data != "":
        r = float(data)
    return r
# fodir = '/u/cs401/Wordlists/'
fodir = './wordlists/'

fptr_luxiaodi = open(fodir+'First-person')
firstP_luxiaodi = fptr_luxiaodi.readlines()
firstP_luxiaodi = [x.lower().strip() for x in firstP_luxiaodi]
fptr_luxiaodi.close()

fptr_luxiaodi = open(fodir+'Second-person')
secP_luxiaodi = fptr_luxiaodi.readlines()
secP_luxiaodi = [x.lower().strip() for x in secP_luxiaodi]
fptr_luxiaodi.close()

fptr_luxiaodi = open(fodir+'Third-person')
thirdP_luxiaodi = fptr_luxiaodi.readlines()
thirdP_luxiaodi = [x.lower().strip() for x in thirdP_luxiaodi]
fptr_luxiaodi.close()

fptr_luxiaodi = open(fodir+'Slang')
slang_luxiaodi = fptr_luxiaodi.readlines()
slang_luxiaodi = [x.lower().strip() for x in slang_luxiaodi]
fptr_luxiaodi.close()

puncStr_luxiaodi = "".join(string.punctuation)

# \/([A-Z.,])([\w]+|\B)
# first second third person pronouns
firstPerRe_luxiaodi = '|'.join(["( |^)" + firP + "\/" for firP in firstP_luxiaodi])
secPerRe_luxiaodi = '|'.join(["( |^)" + secP + "\/" for secP in secP_luxiaodi])
thirdPerRe_luxiaodi = '|'.join(["( |^)" + thirdP + "\/" for thirdP in thirdP_luxiaodi])
# slang
slangRe_luxiaodi = '|'.join(["( |^)" + slang + "\/" for slang in slang_luxiaodi])

# first sec thrid person regex compile
firPerPattern_luxiaodi = re.compile(firstPerRe_luxiaodi)
secPerPattern_luxiaodi = re.compile(secPerRe_luxiaodi)
thirPerPattern_luxiaodi = re.compile(thirdPerRe_luxiaodi)
# coordinating conjuctions
CCPattern_luxiaodi = re.compile("\/CC" + "($| )")
# past tense
vbdPattern_luxiaodi = re.compile("\/VBD" + "($| )")

futurePattern_luxiaodi = re.compile("(^| )" + "('ll\/|will\/|gonna\/|going\/VBG to\/TO \w\/VB)") 
# commas
commaPattern_luxiaodi = re.compile("(^| )" + "\,\/")
# multiple punctuation
multiPuncPattern_luxiaodi = re.compile("((| )[" + puncStr_luxiaodi + "]\/[" 
                                                        + puncStr_luxiaodi + "](| )){2,}") 
# common nouns
commonNounPattern_luxiaodi = re.compile("\/" + "(NNS|NN)" + "($| )")
# proper nouns
properNounPattern_luxiaodi = re.compile("\/" + "(NNPS|NNP)" + "($| )")
# adverb pattern
adverbPattern_luxiaodi = re.compile("\/" + "(RBR|RBS|RB)" + "($| )")

whPattern_luxiaodi = re.compile("\/" + "(WDT|WP|WP\$|WRB)" + "($| )")
# slang pattern
slangPattern_luxiaodi = re.compile(slangRe_luxiaodi)
upwordsPattern_luxiaodi = re.compile("( |^)[A-Z]{3,}\/")
notPuncOnlyPattern_luxiaodi = re.compile("[^\s\/]{0,}" + "[0-9a-zA-Z]" + "[^\s\/]{0,}\/")

bGL_luxiaodi = {line[1]: 
        [strToFloat_luxiaodi(line[3]), strToFloat_luxiaodi(line[4]), 
        strToFloat_luxiaodi(line[5])] 
        for line in csv.reader(
            open(fodir + 'BristolNorms+GilhoolyLogie.csv', "r"), delimiter=',') 
                        if line[1] != "WORD"}
warringer_luxiaodi = {line[1]: 
        [strToFloat_luxiaodi(line[2]), strToFloat_luxiaodi(line[5]), 
                strToFloat_luxiaodi(line[8])] for line in csv.reader(
                    open(fodir + 'Ratings_Warriner_et_al.csv', "r"), delimiter=',') 
                        if line[1] != "Word"}

def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    feats = np.zeros(174)
    # num of first person pronouns
    firstPerPronouns = firPerPattern_luxiaodi.findall(comment)
    feats[0] = len(firstPerPronouns)
    # num of sec person pronouns
    secPerPronouns = secPerPattern_luxiaodi.findall(comment)
    feats[1] = len(secPerPronouns) 
    # Number of third-person pronouns
    thirdPerPron = thirPerPattern_luxiaodi.findall(comment)
    feats[2] = len(thirdPerPron)
    # 3. Number of coordinating conjunctions
    CCAll = CCPattern_luxiaodi.findall(comment)
    feats[3] = len(CCAll)
    # 4. Number of past-tense verbs
    pastTenseVbs = vbdPattern_luxiaodi.findall(comment)
    feats[4] = len(pastTenseVbs)
    # 5. Number of future-tense verbs
    furtureTenseVbs = futurePattern_luxiaodi.findall(comment)
    feats[5] = len(furtureTenseVbs)
    # 6. Number of commas
    feats[6] = len(commaPattern_luxiaodi.findall(comment))
    # 7. Number of multi-character punctuation tokens
    feats[7] = len(multiPuncPattern_luxiaodi.findall(comment))
    # 8. Number of common nouns
    feats[8] = len(commonNounPattern_luxiaodi.findall(comment))
    # 9. Number of proper nouns
    feats[9] = len(properNounPattern_luxiaodi.findall(comment))
    # Number of adverbs
    adverbs = adverbPattern_luxiaodi.findall(comment)
    feats[10] = len(adverbs)
    # Number of wh- words
    feats[11] = len(whPattern_luxiaodi.findall(comment))
    # Number of slang acronyms
    feats[12] = len(slangPattern_luxiaodi.findall(comment))
    # Number of words in uppercase (≥ 3 letters long)
    upperWords = upwordsPattern_luxiaodi.findall(comment)
    feats[13] = len(upperWords)
    # Average length of sentences, in tokens
    sentences = comment.strip().split('\n')
    senCount = len(sentences)
    totalLen = 0
    for sent in sentences:
        sent = sent.split(" ")
        while "" in sent:
            sent.remove("")
        # tokenNum += len(sent)
        # # charlen total
        # charCount += len("".join(sent))
        totalLen += len(sent)
    feats[14] = totalLen/float(senCount)
    # Average length of tokens, excluding punctuation-only tokens, in characters
    trueTokens = notPuncOnlyPattern_luxiaodi.findall(comment)
    tokenNum = len(trueTokens)
    charCount = len("".join(trueTokens)) - tokenNum
    if tokenNum > 0:
        feats[15] = charCount/float(tokenNum)
    else:
        feats[15] = 0
    # Number of sentences.
    feats[16] = senCount


    tokens = comment.split(" ")
    numWords = [0,0]
    aoa = []
    img = []
    fam = []
    warringerStats = [[], [], []]

    for token in tokens:
        word = token.rsplit("/")
        # print(word)
        word = word[0]
        if word in bGL_luxiaodi:
            stats = bGL_luxiaodi[word]
            aoa.append(stats[0])
            img.append(stats[1])
            fam.append(stats[2])
            numWords[0] += 1
        if word in warringer_luxiaodi:
            stats = warringer_luxiaodi[word]
            warringerStats[0].append(stats[0])
            warringerStats[1].append(stats[1])
            warringerStats[2].append(stats[2])
            numWords[1] += 1
    # Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms 
    feats[17] = np.mean(aoa) if len(aoa) > 0 else 0 
    # Average of IMG from Bristol, Gilhooly, and Logie norms
    feats[18] = np.mean(img) if len(img) > 0 else 0 
    # Average of FAM from Bristol, Gilhooly, and Logie norms
    feats[19] = np.mean(fam) if len(fam) > 0 else 0 
    # Standard deviation of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    feats[20] = np.std(aoa) if len(aoa) > 0 else 0 
    # Standard deviation of IMG from Bristol, Gilhooly, and Logie norms
    feats[21] = np.std(img) if len(img) > 0 else 0 
    # Standard deviation of FAM from Bristol, Gilhooly, and Logie norms
    feats[22] = np.std(fam) if len(fam) > 0 else 0 
    # Average of V.Mean.Sum from Warringer norms
    feats[23] = np.mean(warringerStats[0]) if len(warringerStats[0]) > 0 else 0 
    # Average of A.Mean.Sum from Warringer norms
    feats[24] = np.mean(warringerStats[1]) if len(warringerStats[1]) > 0 else 0 
    # Average of D.Mean.Sum from Warringer norms
    feats[25] = np.mean(warringerStats[2]) if len(warringerStats[2]) > 0 else 0 
    # Standard deviation of V.Mean.Sum from Warringer norms
    feats[26] = np.std(warringerStats[0]) if len(warringerStats[0]) > 0 else 0 
    # Standard deviation of A.Mean.Sum from Warringer norms
    feats[27] = np.std(warringerStats[1]) if len(warringerStats[1]) > 0 else 0 
    # Standard deviation of D.Mean.Sum from Warringer norms
    feats[28] = np.std(warringerStats[2]) if len(warringerStats[2]) > 0 else 0 
    return feats
    # 29-172. LIWC/Receptiviti features
    # 173 alt, center, right, left
def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))

    cNpy = np.load(fodir + "../feats/Center_feats.dat.npy")
    lNpy = np.load(fodir + "../feats/Left_feats.dat.npy")
    rNpy = np.load(fodir + "../feats/Right_feats.dat.npy")
    aNpy = np.load(fodir + "../feats/Alt_feats.dat.npy")

    npyData = [cNpy, lNpy, rNpy, aNpy]

    cFptr = open(fodir + "../feats/Center_IDs.txt", "r")
    lFptr = open(fodir + "../feats/Left_IDs.txt", "r")
    rFptr = open(fodir + "../feats/Right_IDs.txt", "r")
    aFptr = open(fodir + "../feats/Alt_IDs.txt", "r")
    cLines = [x.strip() for x in cFptr.readlines() if x]
    lLines = [x.strip() for x in lFptr.readlines() if x]
    rLines = [x.strip() for x in rFptr.readlines() if x]
    aLines = [x.strip() for x in aFptr.readlines() if x]
    ids = [cLines, lLines, rLines, aLines]
    cFptr.close()
    lFptr.close()
    rFptr.close()
    aFptr.close()

    caToNum = {'Center': 1, 'Left': 0, 'Right': 2, 'Alt': 3}
    # loop through input
    for i in range(len(data)):
        line = data[i]
        dataID = line["id"]
        featsExtracted = extract1(line["body"])
        print(featsExtracted[:29])
        for j in range(len(ids)):
            if dataID in ids[j]:
                rowNum = ids[j].index(dataID)
                print("in here")
            else:
                feats[i] = featsExtracted
                continue
            featsExtracted[29:-1] = npyData[j][rowNum]
            break
        feats[i] = featsExtracted
        feats[i][173] = caToNum[line["cat"]]
        # print(feats[i])
    np.savez_compressed( args.output, feats)

    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)

