### 3. Polynomial Regression
# 공부 시간에 따른 시험 점수 (우등생)
from optparse import Values
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('C:/Users/wndud\Desktop/나도코딩/활용편7 머신러닝/PolynomialRegressionData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


## 3-1. 단순 선형 회귀 (simple Linear Regressgion)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X, y) # 전체 데이터로 학습

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ데이터 시각화(전체)ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

# plt.scatter(X, y, color='blue') # 산정도
# plt.plot(X, reg.predict(X), color='green') # 선 그래프
# plt.title('Score by hours (genius)')
# plt.xlabel('hours') # x 축 이름
# plt.ylabel('score') # y 축 이름
# # plt.show()

print(reg.score(X, y)) # 전체 데이터를 통한 모델 평가   0.8169296513411765


## 3-2. 다항 회귀 (Polynomial Regression)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4) # 2차
X_poly = poly_reg.fit_transform(X)      # fit : 새롭게 만들 피처 찾음 / transform : 실제로 데이터를 만드는 역할을 한다.
# print(X_poly[:5])                       # x -> [x^0, x^1, x^2] -> x가 3이라면 [1, 3, 9]으로 변환  / print(poly_reg.get_feature_names_out())

# print(X[:5])

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y) # 변환된 X 와 y를 가지고 모델 생성 (학습)

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ데이터 시각화(변환된 X와 y)ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

# plt.scatter(X, y, color='blue')
# plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color='green')
# plt.title('Score by hours (genius)')
# plt.xlabel('hours') # x 축 이름
# plt.ylabel('score') # y 축 이름
# # plt.show()

X_range = np.arange(min(X), max(X), 0.1) # X의 최소값에서 최대값까지의 범위를 0.1단위로 잘라서 데이터를 생성
X_range = X_range.reshape(-1, 1)            # row 개수는 자동으로 계산, column 개수는 1개


plt.scatter(X, y, color='blue')
plt.plot(X_range, lin_reg.predict(poly_reg.fit_transform(X_range)), color='green')
plt.title('Score by hours (genius)')
plt.xlabel('hours') # x 축 이름
plt.ylabel('score') # y 축 이름
# plt.show()


### 공부 시간에 따른 시험 성적 예측
print(reg.predict([[2]]))  # 2시간을 공부했을 때 선형 회귀 모델 예측

print(lin_reg.predict(poly_reg.fit_transform([[2]])))    # 2시간 공부했을 때 다항 회귀 모델의 예측

print(lin_reg.score(X_poly, y)) # 0.9782775579000045
