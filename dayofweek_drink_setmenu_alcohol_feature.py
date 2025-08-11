import pandas as pd
import numpy as np

file_path = r"C:\Users\minseo\lg\train\train.csv"
# file_path_2 = r"C:\Users\minseo\lg\test"

df = pd.read_csv(file_path, encoding='utf-8')  # 글자 깨지면 cp949

#dayofweek
df['영업일자'] = pd.to_datetime(df['영업일자'])
df['dayofweek'] = df['영업일자'].dt.dayofweek

drink_keywords = ['콜라', '스프라이트', '제로콜라', '자몽리치에이드', '애플망고 에이드', '핑크레몬에이드', '아메리카노', 
                  '식혜', '메밀미숫가루', '아메리카노', '카페라떼', '복숭아 아이스티','샷 추가',
                  '생수']

alcohol_keywords = ['Gls.Sileni', 'Gls.미션 서드', '미션 서드 카메르네 쉬라', '하이네켄', '막걸리',
                    '와인', '버드와이저', '스텔라', '하이볼', '잭 애플 토닉', '참이슬', '소주', '처음처럼', 
                    '카스', '테라', '칵테일', 'Cass']

set_keywords = ['정식']

df['is_drink'] = df['영업장명_메뉴명'].apply(
    lambda x: 1 if any(keyword in str(x) for keyword in drink_keywords) else 0
)

df['is_alcohol'] = df['영업장명_메뉴명'].apply(
    lambda x: 1 if (
        any(keyword in str(x) for keyword in alcohol_keywords)
        and '컵' not in str(x)
    ) else 0
)

df['is_set_menu'] = df['영업장명_메뉴명'].apply(
    lambda x: 1 if (
        any(keyword in str(x) for keyword in set_keywords)
    ) else 0
)


#메뉴명 컬럼이 없고 '영업장명_메뉴명'만 있는 경우 분리하기
if '메뉴명' not in df.columns and '영업장명_메뉴명' in df.columns:
    # '영업장명_메뉴명'-> 첫 '_' 뒤를 메뉴명으로 사용
    df['메뉴명'] = df['영업장명_메뉴명'].astype(str).str.split('_', n=1).str[-1]

if '영업장명' not in df.columns and '영업장명_메뉴명' in df.columns:
    #영업장명_메뉴명 
    split = df['영업장명_메뉴명'].astype(str).str.split('_', n=1, expand=True)
    df['영업장명'] = split[0]


#매출수량이 문자열이면 숫자로 변환 
df['매출수량'] = pd.to_numeric(df['매출수량'], errors='coerce')



#3. 영업장별 평균 대비
df['avg_store'] = df.groupby('영업장명')['매출수량'].transform('mean')
df['is_popular_menu_store'] = (df['매출수량']> df['avg_store']).astype(int)

#평균 컬럼 제거하기 
df = df.drop(columns=['avg_store'])

# print(df[['영업일자', 'dayofweek', '영업장명_메뉴명', 'is_drink', 'is_alcohol', 'is_set_menu','seasonal_index', 'seasonal_average','is_popular_menu']].head(8000))

output_path = r"C:\Users\minseo\lg\train_with_features_4.csv"

df.to_csv(output_path, index=False, encoding='utf-8-sig')
# df_filtered = df[(df['is_alcohol'] == 1)]
# print(df_filtered[['영업일자', '요일', '영업장명_메뉴명', 'is_drink', 'is_alcohol']])