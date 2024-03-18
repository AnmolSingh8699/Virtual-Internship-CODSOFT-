import pandas as pd
from sklearn.linear_model import LinearRegression as LR
from sklearn.svm import SVC
csv_file_path = "C:/Users/Admin/Desktop/data.csv"  
df1 = pd.read_csv(csv_file_path)
print(df1)
model = SVC()
model.fit(df1[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]], df1["Species"])
SVC()
prediction = model.predict([[5.2,4.6,5.1,6.0]])
print(prediction)
