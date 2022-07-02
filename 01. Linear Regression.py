# 1. Linear Regressin
# 공부 시간에 따른 시험 점수
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('LinearRegressionData.csv')
# print(dataset)
X = dataset.iloc[:, :-1].values # 처음부터 마지막 컬럼 직전까지의 데이터 (독립 변수 - 원인)
y = dataset.iloc[:, -1].values # 마지막 컬럼 데이터 (종속 변수 - 결과)
# print(X)
# print(Y)

from sklearn.linear_model import LinearRegression
reg = LinearRegression() # 객체 생성
reg.fit(X, y) # fit - 훈련시킴 -> 학습 (모델 생성)

y_pred = reg.predict(X) # X에 대한 예측 값
# print(y_pred)

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ데이터시각화ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# plt.scatter(X, y, color='blue')     # 산정도
# plt.plot(X, y_pred, color='green')      # 선 그래프
# plt.title('Score by hours')     # 제목
# plt.xlabel('hours')     # x축 이름
# plt.ylabel('score')     # y축 이름
# plt.show()

print('9시간 공부했을 때 예상 점수 : ', reg.predict([[9]])) #[9], [8], [7] 차원조건 동일해야 함.
print(reg.coef_)     # 기울기 m값 
print(reg.intercept_) # y절편 b값






