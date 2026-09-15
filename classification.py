import pandas as pd
#from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
data=pd.read_csv("diabetes.csv")

# print(data.head(10))
# print(data.info())
# stats =(data.describe())

# profile= ProfileReport(data,title="Diabetes report",explorative=True)
# profile.to_file("diabetes_report.html")

target="Outcome"
x= data.drop(target,axis=1)
y=data[target]

#Split data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
#Lấy 80% train cũ, tiếp tục lấy 25% của nó làm validation và 75% còn lại tiếp tục làm training.
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.25,random_state=100)

# print("X_train:", x_train.shape)
# print("y_train:", y_train.shape)
# print("X_val:", x_val.shape)
# print("y_val:", y_val.shape)
# print("X_test:", x_test.shape)
# print("y_test:", y_test.shape)
'''
FIT       = learn
TRANSFORM = embrace + adapt
FIT_TRANSFORM = Hybrid 
'''
#Preprocess data
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train) # (x-mean)/standard
# Có thể ghi nhanh thành x_train=scaler.fit_transform(x_train)
x_val=scaler.transform(x_val)# chỉ được đo chứ không được fit để tránh data leakage
x_test=scaler.transform(x_test) #transform chỉ thỏa nếu đã fit ít nhất 1 lần

# Pick model()
model1= SVC()
model2=RandomForestClassifier()

# Train the selected model
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)

print("Dự đoán dựa trên tập x_test")
y_predict1=model1.predict(x_test)
y_predict2=model2.predict(x_test)
for pred1,pred2,actual in zip(y_predict1,y_predict2,y_test.values):
    print("Predicted SVC:{} || Predicted RandomForestClassifier:{} || Actual value:{}".format(pred1,pred2,actual))

print("Dự đoán dựa trên tập x_val")
y_val_predict1=model1.predict(x_val)
y_val_predict2=model2.predict(x_val)
result=pd.DataFrame({"Actual":y_val.values,"SVC":y_val_predict1,"RandomForestClassifier":y_val_predict2})
print(result)

print(f"Accuracy score:{accuracy_score(y_test,y_predict1)*100}%")
print(f"Precision score:{precision_score(y_test,y_predict1)*100}%")
print(f"Recall score:{recall_score(y_test,y_predict1)*100}%")
print(f"F1 score:{f1_score(y_test,y_predict1)*100}%")

print(confusion_matrix(y_test,y_predict1))
cm = confusion_matrix(y_test, y_predict1)
print("TN =", cm[0, 0])
print("FP =", cm[0, 1])
print("FN =", cm[1, 0])
print("TP =", cm[1, 1])
print(confusion_matrix(y_test,y_predict2))
cm = confusion_matrix(y_test, y_predict2)
print("TN =", cm[0, 0])
print("FP =", cm[0, 1])
print("FN =", cm[1, 0])
print("TP =", cm[1, 1])




