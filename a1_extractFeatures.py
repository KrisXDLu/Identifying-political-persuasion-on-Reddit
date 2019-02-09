import numpy as np
import sys
import argparse
import os
import json
import re



fptr_1001299944 = open(fodir+'First-person')
firstP_1001299944 = fptr_1001299944.readlines()
firstP_1001299944 = [x.lower().strip() for x in firstP_1001299944]
fptr_1001299944.close()

fptr_1001299944 = open(fodir+'Second-person')
secP_1001299944 = fptr_1001299944.readlines()
secP_1001299944 = [x.lower().strip() for x in secP_1001299944]
fptr_1001299944.close()

fptr_1001299944 = open(fodir+'Third-person')
thirdP_1001299944 = fptr_1001299944.readlines()
thirdP_1001299944 = [x.lower().strip() for x in thirdP_1001299944]
fptr_1001299944.close()

fptr_1001299944 = open(fodir+'Slang')
slang_1001299944 = fptr_1001299944.readlines()
slang_1001299944 = [x.lower().strip() for x in slang_1001299944]
fptr_1001299944.close()

puncStr_1001299944 = "".join(string.punctuation)

# \/([A-Z.,])([\w]+|\B)

firstPerRe_1001299944 = '|'.join(["( |^)" + firP + "\/" for firP in firstP_1001299944])
secPerRe_1001299944 = '|'.join(["( |^)" + secP + "\/" for secP in secP_1001299944])
thirdPerRe_1001299944 = '|'.join(["( |^)" + thirdP + "\/" for thirdP in thirdP_1001299944])
slangRe_1001299944 = '|'.join(["( |^)" + slang + "\/" for slang in slang_1001299944])

firPerPattern_1001299944 = re.compile(firstPerRe_1001299944)
secPerPattern_1001299944 = re.compile(secPerRe_1001299944)
thirPerPattern_1001299944 = re.compile(thirdPerRe_1001299944)
slangPattern_1001299944 = re.compile(slangRe_1001299944)
CCPattern_1001299944 = re.compile("\/CC" + "( |$)")
vbdPattern_1001299944 = re.compile("\/VBD" + "( |$)")
futurePattern_1001299944 = re.compile("( |^)" + "('ll\/|(will\/|gonna\/|going\/VBG to\/TO \w\/VB)") 
commaPattern_1001299944 = re.compile("( |^)" + "\,\/")
multiPuncPattern_1001299944 = re.compile("(( |)[" + puncStr_1001299944 + "]\/[" 
                                                        + puncStr_1001299944 + "]( |)){2,}") 
commonNounPattern_1001299944 = re.compile("\/" + "(NNS|NN)" + "( |$)")
properNounPattern_1001299944 = re.compile("\/" + "(NNPS|NNP)" + "( |$)")
adverbPattern_1001299944 = re.compile("\/" + "(RBR|RBS|RB)" + "( |$)")
whPattern_1001299944 = re.compile("\/" + "(WDT|WP|WP\$|WRB)" + "( |$)")
slangPattern_1001299944 = re.compile(slang_regex_1001299944)
upwordsPattern_1001299944 = re.compile("( |^)[A-Z]{3,}\/")

# bGL_1001299944 = 

def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    print('TODO')
    # TODO: your code here

def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))

    # TODO: your code here

    np.savez_compressed( args.output, feats)

    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)

