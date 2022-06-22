import subprocess

p = subprocess.Popen('pip install -r requirements.txt', stdout=subprocess.PIPE, shell=True)
(output, err) = p.communicate()
p_status = p.wait()
print("Command output : ", output)
print("Command exit status/return code : ", p_status)
print("Installed required libraries")


import importlib.util
import sys

def check_package(name):
    if name in sys.modules:
        print(f"{name!r} already in sys.modules")
    elif (spec := importlib.util.find_spec(name)) is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        print(f"{name!r} has been imported")
    else:
        print(f"Can't find the {name!r} module")
        print(f"Please install {name!r} module using pip")
        return 'Error'
    

if __name__ == "__main__" :
    
    sk = check_package('sklearn')
    if sk == 'Error' :
        raise Exception ('scikit-learn package not installed!\nPlease install "scikit-learn" package using pip')
        
    
    #Importing necessary packages
    import sklearn
    from sklearn.linear_model import LinearRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
    
    #loading train and test files
    trainFile = "train\\News_Train.tsv"
    devFile = "train\\News_Dev.tsv"
    testFile = "test\\News.txt"
    
    #Reading the data
    def readTrainDevFile(fileName):
        wordList = []
        labelList = []
        scoreList = []
        f = open(fileName, encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            parts = line.split("\t")
            word = parts[4]
            label = parts[9]
            score = parts[10]
            wordList.append(word)
            scoreList.append(score)
            labelList.append(label)
        return wordList, scoreList, labelList
    
    #Reading the data
    def readTestFile(fileName):
        wordList = []
        f = open(fileName)
        lines = f.readlines()
        for line in lines:
            parts = line.split("\t")
            word = parts[4]
            wordList.append(word)
        return wordList
    
    
    #Creating feature vectors
    def getVowelCount(word):
        count = 0
        for i in range(0, len(word)):
            c = word[i]
            if c == 'a' or c == 'e' or c == 'i' or c == 'o' or c == 'u':
               count = count + 1
        return count

    def getLengthFeatures(word):
        wordLength = len(word)
        vowelCount = getVowelCount(word)
        consonantCount = wordLength - vowelCount
        return wordLength, vowelCount, consonantCount


    #Create the training data
    def getData(wordList):
        data = []
        for i in range(len(wordList)):
            features = getLengthFeatures(wordList[i])
            data.append(features)
        return data

    
    trainWordList, trainScoreList, trainLabelList = readTrainDevFile(trainFile)
    devWordList, devScoreList, devLabelList = readTrainDevFile(devFile)
    testWordList = readTestFile(testFile)
    
    #Instantiate the classifier
    NB = GaussianNB()
    trainX = getData(trainWordList)
    devX = getData(devWordList)
    testX = getData(testWordList)
    trainY = trainLabelList
    devY = devLabelList
    trainZ = trainScoreList
    devZ = devScoreList
    
    devLabelPred = NB.fit(trainX, trainY).predict(devX)
    
    with open("Result\\DevLabelPred.txt","w+") as result_file:
        for i in devLabelPred:
            print(i, file=result_file)
    result_file.close()
    
    testLabelPred = NB.fit(trainX, trainY).predict(testX)
    
    with open("Result\\TestLabelPred.txt","w+") as result_file:
        for i in testLabelPred:
            print(i, file=result_file)
    result_file.close()
    
    
    LR = LinearRegression()
    linear = LR.fit(trainX, trainZ)
    
    devScorePred = LR.predict(devX)
    
    with open("Result\\DevScorePred.txt","w+") as result_file:
        for i in devScorePred:
            print(i, file=result_file)
    result_file.close()

    testScorePred = LR.predict(testX)
    
    with open("Result\\TestScorePred.txt","w+") as result_file:
        for i in testScorePred:
            print(i, file=result_file)
    result_file.close()

    DevZ = [float(i) for i in devZ] #  convert with for loop  
    
    with open("Result\\DevMetrics.txt","w+") as result_file:
        print("\n\n--------------------Dev Metric Analysis---------------------\n",file=result_file)
        print("\nAccuracy score: ", str(accuracy_score(devY, devLabelPred)), file=result_file)
        print("\nMean Squared Error: ", mean_squared_error(DevZ, devScorePred), file=result_file)
        print("\nMean Absolute Error: ", mean_absolute_error(DevZ, devScorePred), file=result_file)
        print("\n**********************************************************************",file=result_file)  
    result_file.close()
    
    print("-----------------------------Dev Metric Analysis--------------------------------")
    print("Accuracy score: ", str(accuracy_score(devY, devLabelPred)))

    print("Mean Squared Error: ", mean_squared_error(DevZ, devScorePred))
    print("Mean Absolute Error: ", mean_absolute_error(DevZ, devScorePred))

