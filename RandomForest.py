import pandas as pd
import numpy as np
import sys
import os
import os.path as P
import json

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#user-defined function
import fetchDataFrame
import drawPlt_TrainAndTest
import makeSDT
import resultVisualization
import saveModel

plt.style.use('fivethirtyeight')
def create_parser():
    from optparse import OptionParser
    p = OptionParser("usage: python %prog [options]")
    p.add_option("-d", "--database", type="string", dest="database", default=None, help="Specify database file name")
    p.add_option("-p", "--parameters", type="string", dest="parameters", default=None, help="Specify file name with parameters")
    p.add_option("-t", "--temporary-dir", type="string", dest="temporary_dir", default=None, help="Specify the directory to use for storing temporary files")
    return p

def remove_unnes_field(featuresTest, featuresTrain):
    for test in featuresTest:
        flag = False
        for train in featuresTrain:
            if test == train:
                flag = True
                break
        if flag == False:
            featuresTest = featuresTest.drop(test, axis=1)
    for train in featuresTrain:
        flag = False
        for test in featuresTest:
            if test == train:
                flag = True
                break
        if flag == False:
            featuresTrain = featuresTrain.drop(train, axis=1)

    return featuresTest, featuresTrain

def check_features(featuresTest ,train_features):
    for train in train_features:
        flag = False
        for test in featuresTest:
            if train == test:
                flag = True
        if flag == False:
            return False
    return True
def main():
    # try:
    print("console")
    parser = create_parser()
    print("console")
    opts, args = parser.parse_args()

    database = opts.database if opts.database else None
    if database is None:
        parser.error("error: database not specified")
    if not P.exists(database):
        parser.error("error: database %s does not exist" % database)
    print("Database: %s" % (database))

    parameters = opts.parameters if opts.parameters else None
    if parameters is None:
        parser.error("error: parameters file not specified")
    if not P.exists(parameters):
        parser.error("error: parameters %s does not exist" % parameters)

    #Get the name of directory place in json file
    srtRootDirectoryBuf = os.path.dirname(os.path.abspath(parameters))

    srtRootDirectory = opts.temporary_dir if opts.temporary_dir else srtRootDirectoryBuf
    if not P.isdir(srtRootDirectory):
        parser.error("error: temporary directory %s does not exist" % srtRootDirectory)
    if os.path.exists(srtRootDirectory + "\\OutputImages") == False:
        os.mkdir(srtRootDirectory + "\\OutputImages")
    print("Temporary dir: %s" % (srtRootDirectory))

    print("Parameters: %s" % (parameters))

    file = open(parameters, mode='r')
    json_string = file.read()
    file.close()
    parameter = json.loads(json_string)

    #check possiblity of using pre trained classifier
    if 'usePretrainedClassifier' in parameter and parameter['usePretrainedClassifier'] == True:
        if 'preTrainedClassifierDir' not in parameter:
            print('preTrainedClassifierDir field is not exist')
			return
        if P.exists(P.join(parameter['preTrainedClassifierDir'], 'featureTrain_list.csv')) == False:
            print('featureTrain_list.csv file is not exist')
			return
        if P.exists(P.join(parameter['preTrainedClassifierDir'], 'IDTrain.csv')) == False:
            print('IDTrain.csv file is not exist')
			return
        if P.exists(P.join(parameter['preTrainedClassifierDir'], 'train_features.csv')) == False:
            print('train_features.csv file is not exist')
			return
        if P.exists(P.join(parameter['preTrainedClassifierDir'], 'train_labels.csv')) == False:
            print('train_labels.csv file is not exist')
			return
        if P.exists(P.join(parameter['preTrainedClassifierDir'], 'TrainModel_3levels.joblib')) == False:
            print('TrainModel_3levels.joblib file is not exist')
			return
        if P.exists(P.join(parameter['preTrainedClassifierDir'], 'TrainModel_Full.joblib')) == False:
            print('TrainModel_Full.joblib file is not exist')
			return
    #Fetch the data for testing and training from DB    
    if 'usePretrainedClassifier' not in parameter or parameter['usePretrainedClassifier'] == False:
        featuresTrain = fetchDataFrame.fetchData(parameters, database, False)
        print("featuresTrain\n", featuresTrain)
        print("featuresTrainDescribe\n",featuresTrain.describe())

    featuresTest = fetchDataFrame.fetchData(parameters, database, True)
    
    #Display the testing data and training data
    print("featuresTest\n", featuresTest)    

    #Descriptive statistics for each column
    
    print("featuresTestDescribe\n",featuresTest.describe())
    
    if 'usePretrainedClassifier' not in parameter or parameter['usePretrainedClassifier'] == False:
        #Visualization of testdata and train data
        drawPlt_TrainAndTest.draw(srtRootDirectory, featuresTest, featuresTrain)

    #Data Preparation(One-Hot Encoding)
    if 'usePretrainedClassifier' not in parameter or parameter['usePretrainedClassifier'] == False:
        featuresTrain = pd.get_dummies(featuresTrain)
        print("one-hot encoding featuresTrain\n", featuresTrain)
        
    featuresTest = pd.get_dummies(featuresTest)
    #display test data and train data after one-hot encoding
    
    print("one-hot encoding featuresTest\n", featuresTest)
    
    if 'usePretrainedClassifier' not in parameter or parameter['usePretrainedClassifier'] == False:
        featuresTest, featuresTrain = remove_unnes_field(featuresTest, featuresTrain)
        print("newfeaturesTrain",len(featuresTrain.columns))
        print("newfeaturesTest",len(featuresTest.columns))
    else:
        file = open(P.join(parameter['preTrainedClassifierDir'], 'featureTrain_list.csv'), mode='r')
        feature_string = file.read()
        featureTrain_list = feature_string.split(',')
        if check_features(featuresTest, featureTrain_list) == False:
            print('Not all trained features are used in test data')
            return
    # Labels and IDs are the values we want to predict
    if 'usePretrainedClassifier' not in parameter or parameter['usePretrainedClassifier'] == False:
        labelsTrain = np.array(featuresTrain['_target_'])
        IDTrain = np.array(featuresTrain['ED_ENC_NUM'])
    else:
        labelsTrain = np.genfromtxt(P.join(parameter['preTrainedClassifierDir'], 'train_labels.csv'), delimiter = ',')
        IDTrain = np.genfromtxt(P.join(parameter['preTrainedClassifierDir'], 'IDTrain.csv'), delimiter = ',')

    labelsTest = np.array(featuresTest['_target_'])
    IDTest = np.array(featuresTest['ED_ENC_NUM'])
    print("IDTest", IDTest)

    # Remove the labels from the features
    # axis 1 refers to the columns
    if 'usePretrainedClassifier' not in parameter or parameter['usePretrainedClassifier'] == False:
        featuresTrain= featuresTrain.drop('_target_', axis = 1)
        featuresTrain= featuresTrain.drop('target_binary', axis = 1)
        featuresTrain= featuresTrain.drop('ED_ENC_NUM', axis = 1)

    featuresTest= featuresTest.drop('_target_', axis = 1)
    featuresTest= featuresTest.drop('target_binary', axis = 1)
    featuresTest= featuresTest.drop('ED_ENC_NUM', axis = 1)

    # Saving feature names for later use
    if 'usePretrainedClassifier' not in parameter or parameter['usePretrainedClassifier'] == False:
        featureTrain_list = list(featuresTrain.columns)
  

    featureTest_list = list(featuresTest.columns)

    # Convert to numpy array
    if 'usePretrainedClassifier' not in parameter or parameter['usePretrainedClassifier'] == False:
        featuresTrain = np.array(featuresTrain)
    else:
        featuresTrain = np.genfromtxt(P.join(parameter['preTrainedClassifierDir'], 'train_features.csv'), delimiter = ',')

    featuresTest = np.array(featuresTest)

    print("featuresTrain - numpy array")
    print(featuresTrain)
    print("featuresTest - numpy array")
    print(featuresTest)

    # impute all missing values
    # https://scikit-learn.org/stable/modules/impute.html#multivariate-feature-imputation
    imp = IterativeImputer(max_iter=10, random_state=0)
    imp.fit(featuresTrain)

    if 'usePretrainedClassifier' not in parameter or parameter['usePretrainedClassifier'] == False:
        featuresTrain = np.round(imp.transform(featuresTrain))
        print("featuresTrain - Missing value added")
        print(featuresTrain)
    featuresTest = np.round(imp.transform(featuresTest))
   
    print("featuresTest - Missing value")
    print(featuresTest)

    # raise Exception('halt', 'halt')


    # Set the data for training and testing
    train_features = featuresTrain
    test_features = featuresTest
    train_labels = labelsTrain
    test_labels = labelsTest
    # Display the data for trainning and testing
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    # Improve Model if Necessary
    rf_new = RandomForestRegressor(n_estimators = 100, criterion = 'mse', max_depth = None, min_samples_split = 2, min_samples_leaf = 1)

    if 'usePretrainedClassifier' in parameter and 'preTrainedClassifierDir' in parameter and parameter['usePretrainedClassifier'] == True:
        fullModelPath = P.join(parameter['preTrainedClassifierDir'], 'TrainModel_Full.joblib')
        threeLevelModelPath = P.join(parameter['preTrainedClassifierDir'], 'TrainModel_3levels.joblib')
    else:
        # Save the model in file
        fullModelPath, threeLevelModelPath = saveModel.save(srtRootDirectory, train_features, train_labels, featureTrain_list, IDTrain)

    # Visualizing a Single Decision Tree
    makeSDT.makeFull(fullModelPath, srtRootDirectory, featureTrain_list, train_features, train_labels, test_features, test_labels)
    makeSDT.makeThreeLevel(threeLevelModelPath, srtRootDirectory, featureTrain_list, train_features, train_labels, test_features, test_labels)

    # Visualize the result
    resultVisualization.resultFull(fullModelPath , IDTrain, IDTest, featureTrain_list, train_features, test_features, train_labels, test_labels, srtRootDirectory)
    resultVisualization.resultThreeLevel(threeLevelModelPath , IDTrain, IDTest, featureTrain_list, train_features, test_features, train_labels, test_labels, srtRootDirectory)

    # except:
    #     print("error")
        
if __name__ == "__main__":
    sys.exit(main())
