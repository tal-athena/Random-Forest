# Import tools needed for visualization
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor
import pydot
import numpy as np
from joblib import load
# is_full: true(full), false(three)
def predict(rf, test_features, test_labels, is_full):
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)

    # Calculate the absolute errors
    errors = abs(predictions - test_labels)

    # Print out the mean absolute error (mae)
    print('Mean Absolute Error For Full:', round(np.mean(errors), 2), 'degrees.')

    # Calculate mean absolute percentage error (MAPE)
    print("testlabels", test_labels)
    is_full = False
    for testV in test_labels:
        print("testlabels", testV)
        if testV == 0:
            is_full = True
            break
    if is_full == False:
        mape = 100 * (errors / test_labels)

        # Calculate and display accuracy
        accuracy = 100 - np.mean(mape)
        if is_full == True:
            print('Accuracy For Full:', round(accuracy, 2), '%.')
        if is_full == True:
            print('Accuracy For Three Levels:', round(accuracy, 2), '%.')

# is_full: true(full), false(three)
def makeSDT(rf, srtRootDirectory, featureTrain_list, train_features, train_labels, is_full):
    # Pull out one tree from the forest
    tree = rf.estimators_[5]

    if is_full == True:
        strDotPath = srtRootDirectory + '\\OutputImages\\tree.dot'
        strImgPath = srtRootDirectory + '\\OutputImages\\tree.png'
    if is_full == False:
        strDotPath = srtRootDirectory + '\\OutputImages\\small_tree.dot'
        strImgPath = srtRootDirectory + '\\OutputImages\\small_tree.png'

    # Export the image to a dot file
    export_graphviz(tree, out_file = strDotPath, feature_names = featureTrain_list, rounded = True, precision = 1)

    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file(strDotPath)

    # Write graph to a png file
    graph.write_png(strImgPath); 
    if is_full == True:
        print('The depth of this full tree is:', tree.tree_.max_depth)
    if is_full == False:
        print('The depth of this three levels tree is:', tree.tree_.max_depth)

def makeFull(fullModelPath, srtRootDirectory, featureTrain_list, train_features, train_labels, test_features, test_labels):
    # rf = train(srtRootDirectory, train_features, train_labels)
    rf_full = load(fullModelPath)
    predict(rf_full, test_features, test_labels, True)
    makeSDT(rf_full, srtRootDirectory, featureTrain_list, train_features, train_labels, True)

def makeThreeLevel(threeLevelModelPath, srtRootDirectory, featureTrain_list, train_features, train_labels, test_features, test_labels):
    rf_3level = load(threeLevelModelPath)
    predict(rf_3level, test_features, test_labels, False)
    makeSDT(rf_3level, srtRootDirectory, featureTrain_list, train_features, train_labels, False)
