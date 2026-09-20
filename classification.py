import pandas as pd
import matplotlib.pyplot as plt
# from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay


data=pd.read_csv("diabetes.csv")

# print(data.head(10))
# print(data.info())
# stats =(data.describe())

# profile= ProfileReport(data,title="Diabetes report",explorative=True)
# profile.to_file("Diabetes_report.html")

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
model2=RandomForestClassifier(random_state=100)

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
#So sánh độ chính xác của 2 model
print("Tập Test-SVC")
print(f"Accuracy score:{accuracy_score(y_test,y_predict1)*100}%")
print(f"Precision score:{precision_score(y_test,y_predict1)*100}%")
print(f"Recall score:{recall_score(y_test,y_predict1)*100}%")
print(f"F1 score:{f1_score(y_test,y_predict1)*100}%")

print("Tập Validation-SVC")
print("Accuracy:", accuracy_score(y_val, y_val_predict1))
print("Precision:", precision_score(y_val, y_val_predict1))
print("Recall:", recall_score(y_val, y_val_predict1))
print("F1:", f1_score(y_val, y_val_predict1))

print("Tập Test-Random Forest")
print(f"Accuracy score:{accuracy_score(y_test,y_predict2)*100}%")
print(f"Precision score:{precision_score(y_test,y_predict2)*100}%")
print(f"Recall score:{recall_score(y_test,y_predict2)*100}%")
print(f"F1 score:{f1_score(y_test,y_predict2)*100}%")

print("Tâp Validation-Random Forest")
print("Accuracy:", accuracy_score(y_val, y_val_predict2))
print("Precision:", precision_score(y_val, y_val_predict2))
print("Recall:", recall_score(y_val, y_val_predict2))
print("F1:", f1_score(y_val, y_val_predict2))


ConfusionMatrixDisplay.from_predictions(y_test,y_predict1)
plt.title("SCV confusion matrix")
plt.show()
cm = confusion_matrix(y_test, y_predict1)
print("TN-Số lượng người không bị bệnh và model đoán đúng là không bị bệnh=", cm[0, 0])
print("FP-Số lượng người không mắc bệnh nhưng mô hình dự đoán có bệnh =", cm[0, 1])
print("FN-Số lượng người mắc bệnh nhưng mô hình dự đoán không có bệnh =", cm[1, 0])
print("TP-Số lượng người mắc bệnh và model đoán đúng là có bệnh =", cm[1, 1])
print(confusion_matrix(y_test,y_predict2))
cm = confusion_matrix(y_test, y_predict2)
print("TN-Số lượng người không bị bệnh và model đoán đúng là không bị bệnh=", cm[0, 0])
print("FP-Số lượng người không mắc bệnh nhưng mô hình dự đoán có bệnh =", cm[0, 1])
print("FN-Số lượng người mắc bệnh nhưng mô hình dự đoán không có bệnh =", cm[1, 0])
print("TP-Số lượng người mắc bệnh và model đoán đúng là có bệnh =", cm[1, 1])




