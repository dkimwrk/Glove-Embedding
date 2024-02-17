import csv
import numpy as np
import argparse
import random

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt


def format_to_six_decimals(number):
    return '{:.6f}'.format(number)

def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()


    train=load_tsv_dataset(args.train_input)
    glovedict= load_feature_dictionary(args.feature_dictionary_in)
    test= load_tsv_dataset(args.test_input)
    valid= load_tsv_dataset(args.validation_input)        



    #removing the ones what are not in the dictionary

    ##################### training #######################       

    length= VECTOR_LEN
    result=[]
    trimmedresult=[]
    for line in train: # “hot not sandwich not square”
        sub=0 #vector length if 300 in this case
        count=0
        parsedline= line[1].split()
        #uniquewords= set(parsedline) # “hot not sandwich square”
        for col in range(length): #1,2,3...300
            for unique in parsedline:
                if unique in glovedict:
                    sub+=glovedict[unique][col]
                    count+=1
            sub= sub/count
            result.append(round(sub,6))
            sub=0
            count=0
        
        trimmedresult.append(result)
        result=[]


    #pasting it into the train data
    for row in range(train.shape[0]):
        train[row][1]= np.array(trimmedresult[row])   

    ##################### testing #######################
    result=[]
    trimmedresult=[]
    for line in test: # “hot not sandwich not square”
        sub=0 #vector length if 300 in this case
        count=0
        parsedline= line[1].split()
        #uniquewords= set(parsedline) # “hot not sandwich square”
        for col in range(length): #1,2,3...300
            for unique in parsedline:
                if unique in glovedict:
                    sub+=glovedict[unique][col]
                    count+=1
            sub= sub/count
            result.append(round(sub,6))
            sub=0
            count=0
        
        trimmedresult.append(result)
        result=[]


    for row in range(test.shape[0]):
        test[row][1]=trimmedresult[row]     


    #############validation###################
    result=[]
    trimmedresult=[]
    for line in valid: # “hot not sandwich not square”
        sub=0 #vector length if 300 in this case
        count=0
        parsedline= line[1].split()
        #uniquewords= set(parsedline) # “hot not sandwich square”
        for col in range(length): #1,2,3...300
            for unique in parsedline:
                if unique in glovedict:
                    sub+=glovedict[unique][col]
                    count+=1
            sub= sub/count
            result.append(round(sub,6))
            sub=0
            count=0
        
        trimmedresult.append(result)
        result=[]


    for row in range(valid.shape[0]):
        valid[row][1]=trimmedresult[row]     

    ########## output ##########
    #label\tvalue1\tvalue2\tvalue3\t...valueM\n.

    ########## train ##########
    outtrain = open(args.train_out, 'w')    

    for line in train:
        sub=""
        sub+=str(format_to_six_decimals(line[0]))
        #outtrain.write(str(format_to_six_decimals(line[0])))
        for word in line[1]:
            sub+='\t'
            sub+=str(format_to_six_decimals(word))
            #outtrain.write('\t')
            #outtrain.write(str(format_to_six_decimals(word)))
        sub+='\n'
        outtrain.write(sub)
        #outtrain.write('\n')
        
    outtrain.close()

    ########## test ##########
    outtest = open(args.test_out, 'w')

    for line in test:
        outtest.write(str(format_to_six_decimals(line[0])))
        for word in line[1]:
            outtest.write('\t')
            outtest.write(str(format_to_six_decimals(word)))
        outtest.write('\n')
        
    outtest.close()


    ########## valid ##########
    outvalid = open(args.validation_out, 'w')

    for line in valid:
        outvalid.write(str(format_to_six_decimals(line[0])))
        for word in line[1]:
            outvalid.write('\t')
            outvalid.write(str(format_to_six_decimals(word)))
        outvalid.write('\n')
        
    outvalid.close()





   
