from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor

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
        strPath = srtRootDirectory + '\\OutputImages\\TrainModel_Full.joblib'
    if flag == False:
        strPath = srtRootDirectory + '\\OutputImages\\TrainModel_3levels.joblib'
    dump(rf, strPath)
    return strPath

def save(srtRootDirectory, train_features, train_labels):
    fullModelPath = getModelPath(srtRootDirectory, train_features, train_labels, True)
    threeLevelModelPath = getModelPath(srtRootDirectory, train_features, train_labels, False)
    return fullModelPath, threeLevelModelPath