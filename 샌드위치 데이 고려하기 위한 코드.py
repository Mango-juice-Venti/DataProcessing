# 2023-01-01 ~ 2025-05-31
# 일자 + isHoliday(주말/공휴일) + isSandwich(샌드위치데이) 생성

import pandas as pd

# holidays 미설치 시 자동 설치
try:
    import holidays
except ImportError:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "holidays>=0.56"])
    import holidays

# 1) 날짜 범위
dates = pd.date_range(start="2023-01-01", end="2025-05-31", freq="D")

# 2) 한국 공휴일(2023~2025)
kr_holidays = holidays.KR(years=[2023, 2024, 2025])

# 3) DataFrame 기본 컬럼
df = pd.DataFrame({"일자": dates})
df["연"] = df["일자"].dt.year
df["월"] = df["일자"].dt.month
df["일"] = df["일자"].dt.day
weekday_map = {0:"월", 1:"화", 2:"수", 3:"목", 4:"금", 5:"토", 6:"일"}
df["요일"] = df["일자"].dt.weekday.map(weekday_map)

# 4) isHoliday: 주말(토/일) 또는 한국 공휴일이면 1, 아니면 0
df["isHoliday"] = (
    (df["일자"].dt.weekday >= 5) |           # 토(5), 일(6)
    (df["일자"].isin(kr_holidays))
).astype(int)

# 5) isSandwich: 오늘은 평일(0)이고, 어제/내일이 모두 쉬는 날(1)인 경우 1
df["isSandwich"] = 0
df.loc[
    (df["isHoliday"] == 0) &
    (df["isHoliday"].shift(1) == 1) &
    (df["isHoliday"].shift(-1) == 1),
    "isSandwich"
] = 1

#(선택) 저장
df.to_csv("calendar_with_sandwich_20230101_20250531.csv", index=False, encoding="utf-8-sig")

# 간단 확인
# print(df.head(10))
# print(df.tail(10))
# print(df[df["isSandwich"]==1].head(20))

df


