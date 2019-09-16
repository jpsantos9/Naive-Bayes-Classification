import pandas as pd
import operator

#retieve data from source
data = pd.read_csv("balance-scale.data", names=["CN", "LW", "LD", "RW", "RD"])

#split data into training and test set
data_copy = data.copy()
train_set = data_copy.sample(frac=0.70, random_state=1)
print("//------- Training Set ---------//")
print(train_set.groupby("CN")["LW"].count())

test_set = data_copy.drop(train_set.index)
print("//---------- Test Set ----------//")
print(test_set.groupby("CN")["LW"].count())

#training the NBC model
#get probability of each class
def getProbClass(NBC_class, mylist):
    total = 0
    class_count = 0
    for elem in mylist:
        total+=1
        if elem==NBC_class:
            class_count+=1
    
    return [class_count, class_count/total]

print("\n//---------- Probability of each Classes [B, R, L] ----------//")
className = ["B", "L", "R"]
classProb = {}  #holds the count and probability of each class 
for cn in className:
    classProb[cn] = getProbClass(cn, train_set["CN"].tolist())

print(pd.DataFrame(classProb),"\n")

# print(train_set[["CN", "LW"]])

def getConditionalProb(NBC_class, feature, featureNum, mylist):
    counter = 0
    total = 0
    for index, row in mylist.iterrows():
        if row["CN"]==NBC_class:
            total+=1
            if row[feature]==featureNum:
                counter+=1
    return [counter, counter/total]

#get the conditional probability for each features
featureVal = [1,2,3,4,5]

#conditional probability for LW
lwProb = {}
for cn in className:
    for value in featureVal:
        lwProb[cn, value] = getConditionalProb(cn,"LW",value,train_set)

print("//---------- Conditional Probability of LW ----------//")
print(pd.DataFrame(lwProb), "\n")

#conditional probability for LD
ldProb = {}
for cn in className:
    for value in featureVal:
        ldProb[cn, value] = getConditionalProb(cn,"LD",value,train_set)

print("//---------- Conditional Probability of LD ----------//")
print(pd.DataFrame(ldProb), "\n")

#conditional probability for RW
rwProb = {}
for cn in className:
    for value in featureVal:
        rwProb[cn, value] = getConditionalProb(cn,"RW",value,train_set)

print("//---------- Conditional Probability of RW ----------//")
print(pd.DataFrame(rwProb), "\n")

#conditional probability for RD
rdProb = {}
for cn in className:
    for value in featureVal:
        rdProb[cn, value] = getConditionalProb(cn,"RD",value,train_set)

print("//---------- Conditional Probability of RD ----------//")
print(pd.DataFrame(rdProb), "\n")

#compute for the evidence
featureName = ["LW", "LD", "RW", "RD"]
evidence = {}
for elem in featureName:
    temp = []
    for value in featureVal:
        temp.append(getProbClass(value,train_set[elem].tolist()))
    evidence[elem] = temp

print("//---------- Evidence ----------//")
print(pd.DataFrame(evidence))
# print(getProbClass(1,train_set["LW"].tolist()))

#predict 
def predict(lw, ld, rw, rd):
    className = ["B", "L", "R"]
    given = {}
    denominator = evidence["LW"][lw-1][1]*evidence["LD"][ld-1][1]*evidence["RW"][rw-1][1]*evidence["RD"][rd-1][1]
    for cn in className:
        numerator = lwProb[cn, lw][1]*ldProb[cn, ld][1]*rwProb[cn, rw][1]*rdProb[cn, rd][1]*classProb[cn][1]
        given[cn] = numerator/denominator

    # print(given)
    return max(given, key=given.get)
    
# print(predict(4,5,5,4))

def testAccuracy(mylist):
    total = 0
    counter = 0
    print("\n//---------- errors ---------//")
    print("T  :  P\n-------")
    for index, row in mylist.iterrows():
        predicted = predict(row["LW"], row["LD"], row["RW"], row["RD"])
        truth = row["CN"]
        total+=1
        if truth==predicted:
            counter+=1
        else:
            print(truth, " : ", predicted)
    print("\n//---------- Accuracy of the NBC Model ----------//")
    return counter/total

print(testAccuracy(test_set)*100)

# def showStat():
