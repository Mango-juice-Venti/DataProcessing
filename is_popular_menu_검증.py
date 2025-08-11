#is_popular_ feature의 적합도 분석->
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import koreanize_matplotlib


#csv 불러오기
file_path = r"C:\Users\minseo\lg\train_with_features_4.csv"
df = pd.read_csv(file_path, encoding='utf-8')

# 1. 상관관계 분석
target = '매출수량'
features = ['is_popular_menu_quarter', 'is_popular_menu_menu', 'is_popular_menu_store']


#상관관계 계산
corr = df[[target] + features].corr()

print("타겟과 각 피처의 상관계수")
print(corr[target].sort_values(ascending=False))

#시각화
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("RandomForest 피처 중요도")
print(importance)

best_feature = importance.iloc[0]['feature']
print(f"모델 기준 가장 영향력 있는 피처: {best_feature}")