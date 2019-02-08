import sys
import argparse
import os
import json
import string
import re
import spacy
import html

# indir = '/u/cs401/A1/data/';
indir = './CSC401A1/data/'
nlp_1001299944 = spacy.load('en', disable=['parser', 'ner'])
#[\s\w]
home = './CSC401A1/'
f = open(home +'wordlists/abbrev.english')
abbrev_1001299944 = f.readlines()
abbrev_1001299944 = [x.strip() for x in abbrev_1001299944]
f = open(home+'wordlists/clitics')
clitics_1001299944 = f.readlines()
clitics_1001299944 = [x.strip() for x in clitics_1001299944]
f.close()
f = open(home+'wordlists/StopWords')
stopWords_1001299944 = f.readlines()
stopWords_1001299944 = [x.strip() for x in stopWords_1001299944]
f.close()
puncStr_1001299944 = string.punctuation
puncStr_1001299944 = puncStr_1001299944.replace("'", '')


def preproc1( comment , steps=range(1,11)):

    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''

    modComm = ''
    if 1 in steps:
        # remove all newline char
        modComm = comment.replace('\n', '')
    if 2 in steps:
        # replace html char with ascii eq
        modComm = html.unescape(modComm)
    if 3 in steps:
        #remove all URLs
        modComm = re.sub(r'^https?:\/\/.*[\r\n]*', '', modComm, flags=re.MULTILINE)
    if 4 in steps:
        # split each punctuation
        punc = ""
        for c in puncStr_1001299944:
            if c != '\\':
                punc += c
            else:
                punc += '\\'
        punc.replace('\'', '')
        regexStr = "(\d+[\,\.]\d+|[" + punc + "])+"
        words = modComm.strip().split()
        modComm = ''
        for item in words:
            if item not in abbrev_1001299944:
                add = ' '.join([x.strip() for x in re.split(regexStr, item) if x])
                modComm += add + " "      

    if 5 in steps:
        # split clitics using whitespace
        regexStr = '(' + '|'.join(clitics_1001299944) + ')'
        modComm = ' '.join([x.strip() for x in re.split(regexStr, modComm) if x])
        modComm = "s ' ".join([x.strip() for x in re.split("s'", modComm) if x])

    if 6 in steps:
        #each token is tagged with its part-of-speech
        utt = nlp_1001299944(modComm)
        modComm = ''
        for token in utt:
            if len(token.text) > 0:
                modComm += token.text + "/" + token.tag_ + " "
        modComm = modComm.strip()
    if 7 in steps:
        #remove stop words
        regexStr = "\s((" + '|'.join(stopWords_1001299944) + ")\/[\S]+)"
        re.sub(regexStr, "", ' ' + modComm)
        modComm = modComm.strip()
    if 8 in steps:
        #apply lemmatization using spaCy
        words = modComm.strip().split()
        modComm = ""
        for item in words:
            token, tag = item.rsplit("/", 1)
            utt = nlp_1001299944(token)
            if len(utt) > 0:
                if not utt[0].lemma_.startswith('-') and utt[0].lemma_ != token[0].lower():
                    modComm += utt[0].lemma_
                else:
                    modComm += token
                modComm += "/" + tag + " "
        modComm.strip()
    
    if 9 in steps:
        #add a new line between sentence(endofsentence detection)
        eos = ".!?"
        words = modComm.strip().split()
        modComm = ''
        for w in range(len(words)):
            item = words[w]
            word, tag = item.strip().rsplit("/", 1)
            if len(word)>0 and (word in abbrev_1001299944 or tag in eos ):
                if (w+1) >= len(words):
                    continue
                if words[w+1][0].isupper() and not words[w+1] in abbrev_1001299944:
                    words[w] += '\n'
        modComm = " ".join(words)
    if 10 in steps:
        #convert text to lower case
        words = modComm.strip().split()
        modComm = ''
        for item in words:
            word, tag = item.strip().rsplit("/", 1)
            modComm += word.lower() + "/" + tag + " "
        modComm.strip()
        
    return modComm

def main( args ):
    print(args)
    allOutput = []
    ID = args.ID[0]
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))

            size = len(data)
            # select appropriate args.max lines
            startInd = ID%size
            endInd = startInd + args.max
            filename  = os.path.basename(fullFile)
            # print(size)
            # print(startInd)
            # print(endInd)
            count = 1
            total = endInd-startInd

            for i in range(startInd, endInd):
                # percentage = count/total
                # if str(percentage*10)[2] == '0':
                #     print(str(percentage*100) + "%")
                if count == 100:
                    print(i-startInd)
            # read those lines with something like `j = json.loads(line)`
                comment = json.loads(data[i])
                newCom = {}
            # choose to retain fields from those lines that are relevant to you
                newCom['id'] = comment['id']
            # process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # replace the 'body' field with the processed text
                newCom['body'] = preproc1(comment['body'])
            # add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
                newCom['cat'] = filename
            # append the result to 'allOutput'
                allOutput.append(newCom)
                count+=1;
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":
    # nlp = spacy.load(’en’, disable=[’parser’, ’ner’])
    # utt = nlp_1001299944(u"Go to St. John’s St. John is there.")
    # for token in utt:
    #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
    #     token.shape_, token.is_alpha, token.is_stop)
    # Go go VERB VB  Xx True False
    # to to ADP IN  xx True True
    # St. st. PROPN NNP  Xx. False False
    # John john PROPN NNP  Xxxx True False
    # ’s ’s PART POS  ’x False False
    # St. st. PROPN NNP  Xx. False False
    # John john PROPN NNP  Xxxx True False
    # is be VERB VBZ  xx True True
    # there there ADV RB  xxxx True True
    # . . PUNCT .  . False False

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type = int, help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
        
    main(args)

