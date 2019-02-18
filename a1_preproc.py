import sys
import argparse
import os
import json
import string
import re
import spacy
import html

indir = '/u/cs401/A1/data/';
# indir = './data/'
nlp_1001299944 = spacy.load('en', disable=['parser', 'ner'])
#[\s\w]
home_1001299944 = '/u/cs401/'
# home_1001299944 = './'
f = open(home_1001299944 +'Wordlists/abbrev.english')
abbrev_1001299944 = f.readlines()
abbrev_1001299944 = [x.strip() for x in abbrev_1001299944]
f = open(home_1001299944+'Wordlists/clitics')
clitics_1001299944 = f.readlines()
clitics_1001299944 = [x.strip() for x in clitics_1001299944]
f.close()
f = open(home_1001299944+'Wordlists/StopWords')
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

    modComm = comment
    if 1 in steps:
        # remove all newline char
        modComm = comment.replace('\n', '')
        # print(1)
        # print(modComm)
    if 2 in steps:
        # replace html char with ascii eq
        modComm = html.unescape(modComm)
        # print(2)
        # print(modComm)
    if 3 in steps:
        #remove all URLs
        modComm = modComm.split(" ")
        com = modComm[:]
        for token in com:
            if token.startswith('http') or token.startswith('www'):
                modComm.remove(token)
        modComm = " ".join(modComm)

        # print(3)
        # print(modComm)
    if 4 in steps:
        # split each punctuation
        # punc = ""
        # for c in puncStr_1001299944:
        #     if '\\' not in c:
        #         punc += c
        #     else:
        #         punc += c[1:]        
        # punc.replace('\'', '')
        # regexStr = "(\d+[\,\.]\d+|[" + punc + "])+"
        # words = modComm.strip().split(" ")
        # modComm = ''
        # for item in words:
        #     if item not in abbrev_1001299944:
        #         add = ' '.join([x.strip() for x in re.split(regexStr, item) if x])
        #         modComm += add + " "      
        # modComm = modComm[:-1]
        output = ''
        size = len(modComm)
        for i in range(size):
            # flag for current abbreviation and preabbrevication
            curAbb = 0
            preAbb = 0
            for abbr in abbrev_1001299944:
                if modComm[:i+1].endswith(abbr):
                    curAbb = 1
                    break
                if modComm[:i].endswith(abbr):
                    preAbb = 1
                    break

            if modComm[i] in puncStr_1001299944 and i > 0 and (
                    not modComm[i-1] in puncStr_1001299944 
                    or preAbb == 1) and (
                    modComm[i-1] != " " and curAbb == 0):
                # need to split curr Char(is a punc and not abbr)
                output = output + ' '

                if i < (size - 1) and not modComm[i + 1] in puncStr_1001299944 and modComm[i + 1] != ' ':
                    # punc in the next char and break so no extra space added
                    output = output + modComm[i] + ' '
                    continue
            output += modComm[i]
            if i == 0 and modComm[i] in puncStr_1001299944:
                # first char is a punc split
                output += ' '
        modComm = output.strip()
        # print(4)
        # print(modComm)
    if 5 in steps:
        # split clitics using whitespace
        regexStr = '(' + '|'.join(clitics_1001299944) + ')'
        modComm = ' '.join([x.strip() for x in re.split(regexStr, modComm) if x])
        modComm = "s ' ".join([x.strip() for x in re.split("s'", modComm) if x])
        # print(5)
        # print(modComm)

    if 6 in steps:
        #each token is tagged with its part-of-speech
        utt = nlp_1001299944(modComm)
        output = ''
        for token in utt:
            if token.text != "" and token.text.strip() != "" and token.text:
                output += token.text + "/" + token.tag_ + " "
        modComm = output.strip()
    if 7 in steps:
        #remove stop words
        # print(stopWords_1001299944)
        words = modComm.split(' ')
        result = words[:]
        for token in words:
            # no tag empty string or space
            if not '/' in token:
                if token.lower() in stopWords_1001299944:
                    result.remove(token)
                continue
            # remove stopwords
            word, tag = token.rsplit("/", 1)
            # print(word)
            if word.lower() in stopWords_1001299944:
                result.remove(token)
        modComm = " ".join(result)
        # print(7)
        # print(modComm)

    if 8 in steps:
        #apply lemmatization using spaCy
        words = modComm.strip().split(" ")
        modComm = ""
        for item in words:
            if item == "" or item.strip() == "":
                continue
            token, tag = item.rsplit("/", 1)
            utt = nlp_1001299944(token)
            # print(utt)
            if len(utt) > 0:
                # if token == 'Writing':
                #     print(utt[0].lemma_)
                #     print(token[0].lower())
                if (utt[0].lemma_[0] != "-") and utt[0].lemma_ != token.lower():
                    modComm += utt[0].lemma_
                else:
                    modComm += token
                modComm += "/" + tag + " "
        modComm.strip()
        # print(8)
        # print(modComm)
    
    if 9 in steps:
        #add a new line between sentence(endofsentence detection)
        eos = ".!?"
        words = modComm.split(" ")
        modComm = ''
        for w in range(len(words)):
            item = words[w]
            if len(item) > 0 and len(item.strip()) > 0 and item:
                word, tag = item.rsplit("/", 1)
                if len(word)>0 and (word in abbrev_1001299944 or tag in eos ):
                    if (w+1) >= len(words):
                        # words[w] += '\n'
                        continue
                    if len(words[w+1]) == 0 or len(words[w+1].strip()) == 0:
                        continue
                    # print(words[w] + ":" + words[w+1])
                    if words[w+1][0].isupper() and not words[w+1] in abbrev_1001299944:
                        words[w] += '\n'
        modComm = " ".join(words)
        # print(9)
        # print(modComm)
    if 10 in steps:
        #convert text to lower case
        words = modComm.split(' ')
        modComm = ''
        for item in words:
            if '/' in item:
                word, tag = item.rsplit("/", 1)
                modComm += word.lower() + "/" + tag + " "
                continue
            if item.strip() != "":
                modComm += item.lower() + " "
        modComm.strip()
        # print(10)
        # print(modComm)
        
    return modComm

def main( args ):
    # print(args)
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
                if count % 50 == 0:
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
    
    # print(preproc1("where have you been???\nWriting tests\n Prof. Col. ASKED me to go to http://www.naodi.com that's cool", steps=range(1, 11)))


