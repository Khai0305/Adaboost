import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

letters_cg = pd.read_csv('letters_CG.csv')
X = letters_cg[['x-box','y-box','width','high','onpix','x-bar','y-bar','x2bar','y2bar','xybar','x2ybr','xy2br','x-ege','xegvy','y-ege','yegvx']]
y = letters_cg['Class']

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3311258278145696)
# traning = 500
from sklearn.ensemble import AdaBoostClassifier


# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)

# Train Adaboost Classifer
model1 = abc.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = model1.predict(X_test)
print("KQ cho 20 điểm test :")
print("Label thật của data: ", *y_pred[20:40])
print("Đầu ra dự đoán: ", *y_test[20:40])

from sklearn.metrics import accuracy_score
# calculate and print model accuracy
print("AdaBoost Classifier Model Accuracy:", accuracy_score(y_test, y_pred))