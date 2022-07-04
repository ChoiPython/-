# # 1. Linear Regressin
# # 공부 시간에 따른 시험 점수
# import matplotlib.pyplot as plt
# import pandas as pd

# dataset = pd.read_csv('LinearRegressionData.csv')
# # print(dataset)
# X = dataset.iloc[:, :-1].values # 처음부터 마지막 컬럼 직전까지의 데이터 (독립 변수 - 원인)
# y = dataset.iloc[:, -1].values # 마지막 컬럼 데이터 (종속 변수 - 결과)
# # print(X)
# # print(Y)

# from sklearn.linear_model import LinearRegression
# reg = LinearRegression() # 객체 생성
# reg.fit(X, y) # fit - 훈련시킴 -> 학습 (모델 생성)

# y_pred = reg.predict(X) # X에 대한 예측 값
# # print(y_pred)

# #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ데이터시각화ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# # plt.scatter(X, y, color='blue')     # 산정도
# # plt.plot(X, y_pred, color='green')      # 선 그래프
# # plt.title('Score by hours')     # 제목
# # plt.xlabel('hours')     # x축 이름
# # plt.ylabel('score')     # y축 이름
# # plt.show()

# print('9시간 공부했을 때 예상 점수 : ', reg.predict([[9]])) #[9], [8], [7] 차원조건 동일해야 함.
# print(reg.coef_)     # 기울기 m값 
# print(reg.intercept_) # y절편 b값

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ데이트 세트 분리ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('LinearRegressionData.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split    # 튜플 형태로 4개 분리 훈련2 테스트 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # 훈련 80 : 테스트 20 으로 분리

# print(X, len(X))  # 전체 데이터

# print(X_train, len(X_train)) # 훈련 세트 X, 개수

# print(X_test, len(X_test)) # 테스트 세트 X, 개수

# print(y, len(y))

# print(y_train, len(y_train)) # 훈련 세트 y, 개수

# print(y_test, len(y_test)) # 테스트 세트 y, 개수

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ분리된 데이터를 통한 모델링ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 

from sklearn.linear_model import LinearRegression
reg = LinearRegression()

reg.fit(X_train, y_train) # 훈련 세트로 학습


#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ데이터 시각화ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# plt.scatter(X_train, y_train, color='blue')     # 산정도
# plt.plot(X_train, reg.predict(X_train), color='green')      # 선 그래프
# plt.title('Score by hours - Train Data')     # 제목
# plt.xlabel('hours')     # x축 이름
# plt.ylabel('score')     # y축 이름
# plt.show()


#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ데이터 시각화ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# plt.scatter(X_test, y_test, color='blue')     # 산정도
# plt.plot(X_train, reg.predict(X_train), color='green')      # 선 그래프
# plt.title('Score by hours - Test Data')     # 제목
# plt.xlabel('hours')     # x축 이름
# plt.ylabel('score')     # y축 이름
# plt.show()

# print(reg.coef_) # 기울기 m 값
# print(reg.intercept_) # y절편 b값

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ모델 평가ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# print(reg.score(X_test, y_test)) # 테스트 세트를 통한 모델 평가

# print(reg.score(X_train, y_train)) # 훈련 세트를 통한 모델 평가









