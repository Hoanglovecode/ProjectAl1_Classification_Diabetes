import pandas as pd
#from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
data=pd.read_csv("diabetes.csv")
# print(data.head(10))
#
# print(data.info())
#
# stats =(data.describe())
target="Outcome"
# profile= ProfileReport(data,title="Diabetes report",explorative=True)
# profile.to_file("diabetes_report.html")
x= data.drop(target,axis=1)
y=data[target]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)



