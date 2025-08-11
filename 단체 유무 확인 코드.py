import pandas as pd
import re

# ===== 경로 설정 =====
file_path = r"C:\Users\user\Desktop\LG AIMERS\코드\train.csv"

# ===== 단체 관련 키워드 =====
GROUP_PATTERNS = [
    r"단체", r"group", r"grp", r"團體|团体",
    r"패밀리", r"가족세트",
    r"BBQ\s*55", r"BBQ55",
    r"코스\s*\(단체\)", r"세트\s*\(단체\)"
]
pattern = re.compile("|".join(GROUP_PATTERNS), re.IGNORECASE)

# ===== 데이터 읽기 =====
df = pd.read_csv(file_path, encoding="utf-8")  # 필요시 cp949로 변경
df['영업일자'] = pd.to_datetime(df['영업일자'])

# ===== 날짜별 단체 여부 판단 =====
# 단체 관련 메뉴만 필터
df_group = df[df['영업장명_메뉴명'].str.contains(pattern)]

# 매출수량이 1 이상인 날짜
dates_with_group = df_group.loc[df_group['매출수량'] > 0, '영업일자'].unique()

# is_group_keyword 컬럼 추가
df['is_group_keyword'] = df['영업일자'].isin(dates_with_group).astype(int)

# ===== 저장 =====
output_path = r"C:\Users\user\Desktop\LG AIMERS\코드\train_with_is_group_keyword.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("생성 완료:", output_path)
