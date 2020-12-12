import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from joblib import load

# flag: true(full), false(three)
def findVar(rf, featureTrain_list, train_features, test_features, train_labels, test_labels, flag):
    # Get numerical feature importances
    importances = list(rf.feature_importances_)

    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(featureTrain_list, importances)]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    strmostImportant1 = feature_importances[0][0]
    strmostImportant2 = feature_importances[1][0]

    # Print out the feature and importances 
    if flag == True:
        [print('Variable For Full: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    if flag == False:
        [print('Variable For Three Levels: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    # New random forest with only the two most important variables
    rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)

    # Extract the two most important features
    important_indices = [featureTrain_list.index(strmostImportant1), featureTrain_list.index(strmostImportant2)]
    train_important = train_features[:, important_indices]
    test_important = test_features[:, important_indices]

    # Train the random forest
    rf_most_important.fit(train_important, train_labels)

    # Make predictions and determine the error
    predictions = rf_most_important.predict(test_important)

    errors = abs(predictions - test_labels)

    # Display the performance metrics
    if flag == True:
        print('Mean Absolute Error For Full:', round(np.mean(errors), 2), 'degrees.')
    if flag == False:
        print('Mean Absolute Error For Three Levels:', round(np.mean(errors), 2), 'degrees.')

    flagE = False
    for testV in test_labels:
        print("testlabels", testV)
        if testV == 0:
            flagE = True
            break
    if flagE == False:
        mape = np.mean(100 * (errors / test_labels))
        accuracy = 100 - mape

        if flag == True:
            print('Accuracy For Full:', round(accuracy, 2), '%.')
        if flag == True:
            print('Accuracy For Three Levels:', round(accuracy, 2), '%.')
    return importances, predictions

# flag: true(full), false(three)
def display(importances, predictions, IDTrain, IDTest, featureTrain_list, train_labels, srtRootDirectory, flag):
    # list of x locations for plotting
    x_values = list(range(len(importances)))

    # Make a bar chart
    plt.bar(x_values, importances, orientation = 'vertical')

    # Tick labels for x axis
    plt.xticks(x_values, featureTrain_list, rotation='vertical')

    # Axis labels and title
    plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances'); 
    plt.savefig(srtRootDirectory + "\\OutputImages\\importantVariable.png", bbox_inches='tight')
    plt.show(block=False)
    plt.close('all')

    # Dataframe with true values and dates
    true_data = pd.DataFrame(data = {'ED_ENC_NUM': IDTrain, '_target_': train_labels})

    # Dataframe with predictions and dates
    predictions_data = pd.DataFrame(data = {'ED_ENC_NUM': IDTest, 'prediction': predictions}) 
    print("predictionData", predictions_data)
    predictions_data = predictions_data.astype(np.int64)
    print("predictionData", predictions_data)
    # if flag == True:
    #     print("For Full:\n", IDTest, "\n", predictions)
    # if flag == False:
    #     print("For Three Levels:\n", IDTest, "\n", predictions)
    if flag == True:
        predictions_data.to_csv(path_or_buf=srtRootDirectory + "\\output_full.csv", index=False)
    if flag == False:
        predictions_data.to_csv(path_or_buf=srtRootDirectory + "\\output_3levels.csv", index=False)

    # Plot the actual values
    plt.plot(true_data['ED_ENC_NUM'], true_data['_target_'], 'b-', label = '_target_')

    # Plot the predicted values
    plt.plot(predictions_data['ED_ENC_NUM'], predictions_data['prediction'], 'ro', label = 'prediction')
    plt.xticks(rotation = '60'); 
    plt.legend()

    # Graph labels
    plt.xlabel('ED_ENC_NUM'); plt.ylabel('Target Score'); plt.title('Actual and Predicted Values');
    if flag == True:
        plt.savefig(srtRootDirectory + "\\OutputImages\\output_full.png", bbox_inches='tight')
    if flag == False:
        plt.savefig(srtRootDirectory + "\\OutputImages\\output_3levels.png", bbox_inches='tight')
    plt.show(block=False)
    plt.close('all')

def resultFull(fullModelPath , IDTrain, IDTest, featureTrain_list, train_features, test_features, train_labels, test_labels, srtRootDirectory):
    rf = load(fullModelPath)
    importances, predictions = findVar(rf, featureTrain_list, train_features, test_features, train_labels, test_labels, True)
    display(importances, predictions, IDTrain, IDTest, featureTrain_list, train_labels, srtRootDirectory, True)

def resultThreeLevel(threeLevelModelPath , IDTrain, IDTest, featureTrain_list, train_features, test_features, train_labels, test_labels, srtRootDirectory):
    rf = load(threeLevelModelPath)
    importances, predictions = findVar(rf, featureTrain_list, train_features, test_features, train_labels, test_labels, False)
    display(importances, predictions, IDTrain, IDTest, featureTrain_list, train_labels, srtRootDirectory, False)