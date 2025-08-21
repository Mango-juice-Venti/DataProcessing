import pandas as pd
import numpy as np
import glob, os

# -----------------------------
# 유틸: 요일 한글 생성 (일관된 매핑)
# -----------------------------
def add_weekday_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['영업일자'] = pd.to_datetime(out['영업일자'])
    out['요일'] = out['영업일자'].dt.dayofweek.map({0:'월',1:'화',2:'수',3:'목',4:'금',5:'토',6:'일'})
    return out

# -----------------------------
# 1) train에서 (메뉴×요일) → weekday_weight “값 그대로” 매핑 생성
#    - train에 weekday_weight가 아직 없다면: 아래 함수 대신
#      너가 기존에 계산한 방법으로 컬럼을 먼저 만들어 저장한 뒤 이 함수를 써.
# -----------------------------
def build_weight_map_from_train_with_existing_weights(df_train_w: pd.DataFrame) -> pd.DataFrame:
    """
    df_train_w: 반드시 '영업장명_메뉴명','영업일자','weekday_weight' 컬럼 포함
    동일 (메뉴×요일)에 대해 여러 행이 있으면 평균(=같은 값이면 그대로)으로 집계
    """
    dfw = add_weekday_col(df_train_w)
    wm = (dfw
          .groupby(['영업장명_메뉴명','요일'], as_index=False)['weekday_weight']
          .mean())  # 동일 값이면 그대로 유지
    return wm

# -----------------------------
# 2) test에 train의 weight “그대로” 붙이기
# -----------------------------
def attach_train_weight(df_any: pd.DataFrame, weight_map: pd.DataFrame) -> pd.DataFrame:
    df = add_weekday_col(df_any)
    out = df.merge(weight_map, on=['영업장명_메뉴명','요일'], how='left')

    # train에 없는 조합은 1.0 (0 금지)
    out['weekday_weight'] = out['weekday_weight'].fillna(1.0)

    # 요일 제거, weight 맨 끝 배치
    out = out.drop(columns=['요일'])
    cols = [c for c in out.columns if c != 'weekday_weight'] + ['weekday_weight']
    return out[cols]

# -----------------------------
# 경로
# -----------------------------
train_path = "./re_data_processed/re_train_09_with_weight.csv"  # ★ train에 이미 weekday_weight가 들어 있는 파일
test_glob = "./re_data_processed/re_test_processed_07/TEST_*_processed.csv"
test_outdir = "./test_with_weight"
os.makedirs(test_outdir, exist_ok=True)

# -----------------------------
# 실행
# -----------------------------
# (A) train에서 실제 수치 매핑 만들기
df_train_w = pd.read_csv(train_path)
# df_train_w는 반드시 'weekday_weight' 컬럼을 포함해야 함
weight_map = build_weight_map_from_train_with_existing_weights(df_train_w)

# (B) train에도 동일 값으로 재부착(검증용 저장)
df_train_out = attach_train_weight(df_train_w, weight_map)
df_train_out.to_csv("./re_train_09_with_weight_synced.csv", index=False, float_format='%.16f')
print("✅ Saved train (synced): ./re_train_09_with_weight_synced.csv")

# (C) test들에 “같은 값” 붙이기
for path in glob.glob(test_glob):
    df_test = pd.read_csv(path)
    df_test_out = attach_train_weight(df_test, weight_map)
    out_name = os.path.basename(path).replace(".csv", "_with_weight.csv")
    out_path = os.path.join(test_outdir, out_name)
    df_test_out.to_csv(out_path, index=False, float_format='%.16f')
    print("✅ Saved test:", out_path)

print("🎯 완료: train에서 나온 정확한 weekday_weight 실수값을 test에도 ‘그대로’ 부착했습니다.")