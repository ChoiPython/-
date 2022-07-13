### 2. Mutiple Linear Regression
### 원 핫 인코딩
from statistics import LinearRegression
import pandas as pd
dataset = pd.read_csv('C:/Users/wndud/Desktop/나도코딩/활용편7 머신러닝/MultipleLinearRegressionData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# print(X)

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ원-핫 인코딩 모듈ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ct - transformers=[('encoder' => 인코딩을 실행하기 위한 명령어, OneHotEncoder(drop='first') => 인코딩을 수행할 클래스 호출(다중공산성을 없애기 위해 drop='first'), [2] => 어떤 데이터로 원-핫인코딩을 수행할 것인지 지정)], remainder = 'passthrough' => 원-핫 인코딩 이외의 데이터에 대한 조정(passthrough -> )
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [2])], remainder='passthrough')
X = ct.fit_transform(X)         # 1 0 - Home / 0 1 - Libray / 0 0 - Cafe
# print(X)            


#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ데이트 세트 분리ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)


#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ학습(다중 선형 회귀)ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

# 예측값과 실제 값 비교(테스트 세트)
y_pred = reg.predict(X_test)
# print(y_pred)       # [ 92.15457859  10.23753043 108.36245302  38.14675204]
# print(y_test)       # [ 90   8 100  38]

print(reg.coef_)        # 기울기 m값 - [-5.82712824 -1.04450647 10.40419528 -1.64200104]
print(reg.intercept_)   # y절편 b값 - 5.365006706544783

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ모델 평가ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
print(reg.score(X_train, y_train))  # 훈련세트      # 0.9623352565265527
print(reg.score(X_test, y_test))    # 테스트 세트   # 0.9859956178877447






