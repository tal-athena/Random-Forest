import pandas as pd
import numpy as np
import json
import sqlite3

def fetchData(jsonFilepath, dbFilepath, isTest): 
    file = open(jsonFilepath,mode='r')
    json_string = file.read()
    file.close()

    parameter = json.loads(json_string)
    strSql = "SELECT [ED_ENC_NUM], [_target_]";
    numericArr = []
    for field in parameter['fields']:
        if field['name'] != "Category" and field['name'] != "ED_ENC_NUM" and field['name'] != "_target_":
            strSql = strSql + ", [" + field['name'] + "] ";                
        if 'appType' in field and field['appType'] == "Numeric":
            numericArr.append(field['name'])
    if isTest == True:
        strSql = strSql + "FROM TEST;"
    else:
        strSql = strSql + "FROM TRAIN;"
    sqlite_file_name = dbFilepath
    conn = sqlite3.connect(sqlite_file_name)

    features = pd.read_sql_query(strSql, conn)
    for numeric in numericArr:
        features[numeric] = pd.to_numeric(features[numeric], errors='coerce')
        features[numeric] = features[numeric].fillna(0)
    
    # load extra dimensions    
    conn.close()
    csv_file_name = None
    if isTest == True:
        if "pathToTestParametersCSV" in parameter: csv_file_name = parameter['pathToTestParametersCSV']
    
    else:
        if "pathToTrainParametersCSV" in parameter: csv_file_name = parameter['pathToTrainParametersCSV']
    
    if csv_file_name != None:
        extra_dimension = pd.read_csv(csv_file_name, sep=';')
        for column in extra_dimension.columns:
            if column != "ED_ENC_NUM":
                extra_dimension[column] = pd.to_numeric(extra_dimension[column], errors = 'coerce')
                extra_dimension[column] = extra_dimension[column].fillna(0)

        features = pd.merge(features, extra_dimension, on='ED_ENC_NUM')
        print('extra-dimension added', features)

    return features