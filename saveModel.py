from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
import numpy
import os.path as P

# flag: true(full), false(three)
def getModelPath(srtRootDirectory, train_features, train_labels, flag):
    # Instantiate model 
    if flag == True:
        rf = RandomForestRegressor(n_estimators= 1000, random_state=42)
    if flag == False:
        rf = RandomForestRegressor(n_estimators=10, max_depth = 3, random_state=42)

    # Train the model on training data (data must not null)
    rf.fit(train_features, train_labels);
    if flag == True:
        strPath = srtRootDirectory + '\\TrainModel_Full.joblib'
    if flag == False:
        strPath = srtRootDirectory + '\\TrainModel_3levels.joblib'
    dump(rf, strPath)
    return strPath

def save(srtRootDirectory, train_features, train_labels, featureTrain_list, IDTrain):
    fullModelPath = getModelPath(srtRootDirectory, train_features, train_labels, True)
    threeLevelModelPath = getModelPath(srtRootDirectory, train_features, train_labels, False)

    numpy.savetxt(P.join(srtRootDirectory, 'train_features.csv'), train_features, delimiter = ',')
    numpy.savetxt(P.join(srtRootDirectory, 'train_labels.csv'), train_labels, delimiter = ',')
    numpy.savetxt(P.join(srtRootDirectory, 'IDTrain.csv'), IDTrain, delimiter = ',')

    file = open(P.join(srtRootDirectory, 'featureTrain_list.csv'), mode='w')
    flag = True
    for feature in featureTrain_list:
        if flag == False:
            file.write(',')
        else:
            flag = False
        file.write(feature)
    file.close()

    return fullModelPath, threeLevelModelPath