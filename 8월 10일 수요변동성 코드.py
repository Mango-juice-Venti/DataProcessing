import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from lightgbm import LGBMRegressor

file_path="C:\\Users\\user\\Desktop\\LG AIMERS\\코드\\train.csv"

df = pd.read_csv(file_path)

menu_stats=(
    df.groupby('영업장명_메뉴명')['매출수량']
    .agg(['mean','std'])
    .reset_index()    
)

menu_stats['수요변동성']=menu_stats['std']/menu_stats['mean']
menu_stats['수요안정성']=1/menu_stats['수요변동성']

menu_stats.rename(columns={'mean':'평균매출수량','std':'표준편차'},inplace=True)

train=train.merge(menu_stats[['수요변동성','수요안정성']], on='매출수량0',how='left')
test=test.merge(menu_stats[['수요변동성','수요안정성']], on='매출수량0',how='left')
