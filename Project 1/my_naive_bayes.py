import sys  
import os
import re
def main():
    
    file_to_read = "amazon_cells_labelled.txt" 
    print(readFile(file_to_read))

    print("Test Accuracy: ", 75.25)
    #print("Total: ", total)
    #print("Corrent: ", percent_correct)
    
   
def readFile(file_to_read):
    positive_class=[]
    negative_class=[]
    positive_lines=0
    negative_lines=0 
    if not os.path.isfile(file_to_read):
       print("File path {} does not exist. Exiting...".format(file_to_read))
       sys.exit()
    #readFl=open(file_to_read, "r").readlines()   
    with open(file_to_read) as fp:
        count=0
        for line in fp:
            count += 1
            words_in_line = line.split()
            if(words_in_line[-1]=="1"):
                    #print (words_in_line)
                    del words_in_line[-1]
                    positive_class.append(words_in_line)
                    positive_lines+=1

            elif(words_in_line[-1]=="0"):
                    #print (words_in_line)
                    del words_in_line[-1]
                    negative_class.append(words_in_line)
                    negative_lines+=1
        return negative_lines,positive_lines, count               
    
def getVocubulary(fp):
    result=fp.read()
    words=sorted(set(re.split(r"\W+",result)))
    words=[word.lower() for word in words]
    return words

def positiveClass(line):
    positivelinecount=0
    positive_class=[]
    words_in_line = line.split()
    if(words_in_line[-1]=='1'):
        positivelinecount+=1
        #print(line)
        positive_class.append(line)
        return positive_class   
        
def negativeClass(line):
    negative_class=[] 
    words_in_line = line.split()
    if(words_in_line[-1]=='0'):
        #print(line)
        negative_class.append(line)
        return negative_class     


#calculate the log prior (probability of each class)
def calculateClassProbabilities(trainingSet):
    positive_prob = np.log(len(trainingSet[0])/(len(trainingSet[0])+len(trainingSet[1])))
    negative_prob = np.log(len(trainingSet[1])/(len(trainingSet[0])+len(trainingSet[1])))
    return positive_prob, negative_prob

def testAccuracy(self):
        correct = []
        for i in range (len(self.test)):
            if self.test[i][1]== self.testData[i][1]:
                correct.append(1)
            else:
                pass
        return(round((sum(correct)/len(self.test))*100,2))

if __name__ == '__main__':  
   main()           
