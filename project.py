import random
import numpy as np
import math
import csv
import sys

#rate the data is trained at
rate=.0000001
rateg=.01

#sigmoid activation function
def sigmoid(x):
    return 1/(1 + math.exp(-x));
#Function to encode data
def encode(data,length):
    encoded = np.matrix(np.zeros([20,1]))
    for word in range(0,length):
        for node in range(0,20):
            data[301,word]=encoded[node,0]
            encoded[node,0]=np.dot(encoder[node],data[:,[word]])
    return encoded;
#Function to decode data
def decode(encoding,length):
    decode_t = np.matrix(np.zeros([20,1]))
    decoded = np.matrix(np.zeros([20,length]))
    for word in range(0,length):
        for node in range(0,20):
            encoding[21]=decode_t[node,0]
            decode_t[node,0]=np.dot(decoder[node],encoding)
            decoded[:,[word]]=decode_t;
    return decoded;
#Function to output a single vector with an output node
def output(decoding):
    out=np.dot(outw,decoding)
    return out;

#Function to train the layers
def train():
    for j in range (0,100):
        for i in range (0,100):
            #initialize gradient matrices
            encoder_gradient = np.matrix(np.zeros([20,302]))
            decoder_gradient = np.matrix(np.zeros([20,22]))
            output_gradient= np.matrix(np.zeros([1,22]))
            word=np.matrix(np.ones([302,1]))
            for dat in range (0,300):
                word[dat,0]=englishVectorFromCSV[i,dat]
            #encoder layer
            encoded = np.matrix(np.zeros([20,1]))
            for node in range(0,20):
                word[301,0]=0;
                encoded[node,0]=np.dot(encoder[node],word)
            encoded_e = np.matrix(np.ones([22,1]))
            for o in range(0,20):
                encoded_e[o,0]=encoded[o,0]
            #decoder layer
            decoded = np.matrix(np.zeros([20,1]))
            for node in range(0,20):
                encoded_e[21,0]=0;
                decoded[node,0]=np.dot(decoder[node],encoded_e)
            decoded_e = np.matrix(np.ones([22,1]))
            for o in range(0,20):
                decoded_e[o,0]=decoded[o,0]
            #output layer
            output=np.dot(outw,decoded_e)
            #output gradient
            out_error=output[0,0]-i*100
            decoded_e=out_error*decoded_e;
            for c in range(0,22):
                output_gradient[0,c]=output_gradient[0,c]+decoded_e[c,0]
            #decoder gradient
            encoded_e=encoded_e*out_error
            for c in range(0,22):
                for r in range(0,20):
                    decoder_gradient[r,c]=decoder_gradient[r,c]+encoded_e[c,0]*outw[r]
            #encoder gradient
            weight_track = np.matrix(np.ones([20,1]))
            for c in range(0,20):
                for r in range(0,20):
                    weight_track[r,0]=weight_track[r,0]+decoder[r,c]*outw[r]
            word=word*out_error
            for c in range(0,302):
                for r in range(0,20):
                    encoder_gradient[r,c]=encoder_gradient[r,c]+word[c,0]*weight_track[r,0]
            #just in case
            #np.savetxt("encoder.csv",encoder, delimiter=",")
            #np.savetxt("decoder.csv",decoder, delimiter=",")
            #np.savetxt("output.csv",outw, delimiter=",")
            #Adjusts weights based on gradients
            for c in range(0,302):
                for r in range(0,20):
                    encoder[r,c]=encoder[r,c]-encoder_gradient[r,c]*rate
            for c in range(0,22):
                for r in range(0,20):
                     decoder[r,c]=decoder[r,c]-decoder_gradient[r,c]*rate
            for c in range(0,22):
                outw[c]=outw[c]-output_gradient[0,c]*rate
            if (j%5==0)&(i%99==0):
                print(output)
    return 1;

def grammarize(words):
    initialize_csv_vector()
    engWords = getEnglishWords()
    spanWords = getSpanishWords()
    word1=words[0]
    word2=words[1]
    indexOfWord1 = engWords.index(word1)
    importedVector1 = getCSVVector(indexOfWord1)
    indexOfWord2 = engWords.index(word2)
    importedVector2 = getCSVVector(indexOfWord2)
    layer1out=np.matrix(np.ones([301,1]))
    for j in range(0,300):
        layer1out[j,0]=importedVector1[j]*grammar[j,0]+importedVector2[j]*grammar[j,1]+grammar[j,2]
    gout=np.dot(grammarout,layer1out)
    gout=sigmoid(gout[0,0])
    gout=int(round(gout));
    return gout;

def traing():
    initialize_csv_vector()
    engWords = getEnglishWords()
    spanWords = getSpanishWords()
    word_list=[["blue","sky"],["sister","is"],["sad","cow"],["ocean","is"],["kind","dolphin"],["how","high"],["favorite","map"],["the","dog"],["favorite","market"],["the","sister"]]
    length=2;
    for k in range (0,51):
        for i in range (0,10):
            words=word_list[i]
            word1=words[0]
            word2=words[1]
            indexOfWord1 = engWords.index(word1)
            importedVector1 = getCSVVector(indexOfWord1)
            indexOfWord2 = engWords.index(word2)
            importedVector2 = getCSVVector(indexOfWord2)
            layer1out=np.matrix(np.ones([301,1]))
            for j in range(0,300):
                layer1out[j,0]=importedVector1[j]*grammar[j,0]+importedVector2[j]*grammar[j,1]+grammar[j,2]
            gout=np.dot(grammarout,layer1out)
            gout=sigmoid(gout[0,0])
            gout=int(round(gout));
            if (i%2==0):
                i=1
            else:
                i=0
            gerror=gout-i
            for j in range(0,301):
                grammarout[j]=grammarout[j]-layer1out[j,0]*rateg*gerror
            for j in range(0,300):
                grammar[j,0]=grammar[j,0]-importedVector1[j]*rateg*gerror*grammarout[j]
                grammar[j,1]=grammar[j,1]-importedVector2[j]*rateg*gerror*grammarout[j]
                grammar[j,2]=grammar[j,2]-rateg*gerror*grammarout[j]
            if (k%50==0):
                print(gout)
    return 1;
    
        
    
##################Initialization of weight vectors#########################
    
#import encoder weights
with open('encoder.csv') as f:
    encoder=np.loadtxt(f, delimiter=',')
f.close()
#import decoder weights
with open('decoder.csv') as f:
    decoder=np.loadtxt(f, delimiter=',')
f.close()
#import output weights
with open('output.csv') as f:
    outw=np.loadtxt(f, delimiter=',')
f.close()
#import decoder weights
with open('grammar.csv') as f:
    grammar=np.loadtxt(f, delimiter=',')
f.close()
#import output weights
with open('grammarout.csv') as f:
    grammarout=np.loadtxt(f, delimiter=',')
f.close()


####################Section to randomize weight vectors#####################

#encoder = np.matrix(np.ones([20,302]))
#for i in range(0,20):
#    for j in range(0,302):
#        encoder[i,j] = random.random()
#        encoder[i,j] = random.random()

#decoder = np.matrix(np.ones([20,22]))
#for i in range(0,20):
#    for j in range(0,22):
#        decoder[i,j] = random.random()
#        decoder[i,j] = random.random()
        
#outw = np.matrix(np.ones([1,22]))
#for i in range(0,22):
#   outw[0,i] = random.random()

#grammar = np.matrix(np.ones([300,3]))
#for i in range(0,300):
    #for j in range(0,3):
        #grammar[i,j] = random.random()
        #grammar[i,j] = random.random()

#grammarout = np.matrix(np.ones([1,301]))
#for i in range(0,301):
   #grammarout[0,i] = random.random()
        

###################Section to initialize static csv's#################
def initialize_csv_vector():
    global englishVectorFromCSV
    englishVectorFromCSV = np.loadtxt(open("englishVectors.csv", "rb"), delimiter=",")

def getEnglishWords():
    words = []
    with open("english.txt") as f:
        for line in f:
            words.append(line.rstrip('\n'))

    return words
def getSpanishWords():
    words = []
    with open("spanish.txt") as f:
        for line in f:
            words.append(line.rstrip('\n'))

    return words

def getCSVVector(index):
    return englishVectorFromCSV[index]


################Main section that runs the code####################
def main(argv):

    #initialize_model()
    initialize_csv_vector()
    engWords = getEnglishWords()
    spanWords = getSpanishWords()
    #trained=train();
    #if trained==1:
            #print("trained! \n")
    #trainedg=traing();
    #if trainedg==1:
    #        print("trained grammar! \n")

    ###Does the translations here###
    print("Type sentences please \n")
    bu=1
    while(bu==1):
        #Gets the sentence
        inputSentence = sys.stdin.readline().rstrip('\n')
        inputWords = inputSentence.lower().split(" ")
        #grammarize the words
        length=len(inputWords)
        i=0
        consecutive=0
        while i<length:
            try:
                indexOfWord = engWords.index(inputWords[i])
                consecutive=consecutive+1
                if consecutive==2:
                    words=[inputWords[i-1],inputWords[i]]
                    gout=grammarize(words)
                    if (gout==1):
                        inputWords[i]=words[0]
                        inputWords[i-1]=words[1]
                        consecutive=0
                    else:
                        consecutive=consecutive-1
                i=i+1
            except ValueError:
                consecutive=0
                i=i+1
                continue
        #Does the translations
        builtString = ""
        for word in inputWords:
            try:
                indexOfWord = engWords.index(word)
            except ValueError:
                builtString += word + " "
                continue
            importedVector = getCSVVector(indexOfWord)
            length=1;
            word=np.matrix(np.ones([302,1]))
            for dat in range (0,300):
                word[dat,0]=importedVector[dat]
            encoded=encode(word,length)
            encoding = np.matrix(np.ones([22,1]))
            for i in range(0,20):
                encoding[i,0]=encoded[i,0]
            #print(encoding)
            decoding=decode(encoding,length)
            decodinge = np.matrix(np.ones([22,1]))
            for i in range(0,20):
                decodinge[i,0]=decoding[i,0]
            #print(decoding)
            out=output(decodinge)/100;
            out=int(round(out[0,0]));
            if out>99:
                out=99
            if out<0:
                out=0
            builtString += spanWords[out] + " "
        print(builtString)
if __name__ == "__main__":
    main(sys.argv[1:])
    
####################saves the weights#####################
np.savetxt("encoder.csv",encoder, delimiter=",")
np.savetxt("decoder.csv",decoder, delimiter=",")
np.savetxt("output.csv",outw, delimiter=",")
np.savetxt("grammar.csv",grammar, delimiter=",")
np.savetxt("grammarout.csv",grammarout, delimiter=",")
