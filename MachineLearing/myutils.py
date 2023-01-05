import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score    
from sklearn.metrics import confusion_matrix

def get_iris(mode=None):
    # 파일 읽기
    iris = pd.read_csv('iris.csv')
    
    # id 데이터 제거
    df = iris.drop(['Id'], axis=1)
    
    # 컬럼명 변경
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    
    # 이진 or 다중
    if mode == 'bin':
        df = df.loc[df['species']!='Iris-virginica']
        
    # 인코딩
    df['species'] = df['species'].map({
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    })
    
    # X, y 분리
    X = df.iloc[:, :-1] # [행, 열]
    y = df.iloc[:, -1] # 원하는 인덱스
    
    # X_train, X_test, y_train, y_test
    return train_test_split(X, y, test_size=0.2, random_state=2022)



def print_score(y_true, y_pred, average='binary'):
    # 정확도
    acc = accuracy_score(y_true, y_pred)
    # 정밀도
    pre = precision_score(y_true, y_pred, average=average)
    # 재현율
    rec = recall_score(y_true, y_pred, average=average)

    print('accuracy:', acc)
    print('precision:', pre)
    print('recall:', pre)
    

def plot_confusion_matrix(y_true, y_pred):
    cfm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(3, 3))
    sns.heatmap(cfm, annot=True, cbar=False, fmt='d')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()