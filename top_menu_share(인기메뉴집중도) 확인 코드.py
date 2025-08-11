import pandas as pd
from pathlib import Path

def ensure_store_col(df, full_col="영업장명_메뉴명", store_col="영업장명"):
    """'영업장명' 컬럼이 없으면 '영업장명_메뉴명'에서 앞 토큰을 추출해 생성"""
    if store_col not in df.columns:
        if full_col not in df.columns:
            raise KeyError(f"'{store_col}'도 없고 '{full_col}'도 없습니다. 실제 컬럼명을 확인해 주세요.\n현재 컬럼: {list(df.columns)}")
        # 앞쪽 첫 토큰(공백/(_) 전까지)을 업장명으로 사용
        # 예: '느티나무 셀프BBQ_BBQ55(단체)' → '느티나무'
        df[store_col] = (
            df[full_col]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.split(" ", n=1, expand=True)[0]
        )
    return df

def add_top_menu_share(df, group_by_cols, menu_col="영업장명_메뉴명",
                       qty_col="매출수량", top_k=1, out_col=None):
    if out_col is None:
        out_col = f"top{top_k}_menu_share"

    # 방어: 필요한 컬럼 체크
    need = set(group_by_cols + [menu_col, qty_col])
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"필요 컬럼 누락: {miss}\n현재 컬럼: {list(df.columns)}")

    # 필요시 음수 제거(반품 등 제외하려면 주석 해제)
    # df = df[df[qty_col] > 0]

    g = (df.groupby(group_by_cols + [menu_col], dropna=False)[qty_col]
           .sum().reset_index())

    total = g.groupby(group_by_cols, dropna=False)[qty_col].sum().rename("group_total")
    g = g.merge(total.reset_index(), on=group_by_cols, how="left")

    g["rank"] = g.groupby(group_by_cols, dropna=False)[qty_col].rank(
        method="first", ascending=False
    )
    topk = (g[g["rank"] <= top_k]
              .groupby(group_by_cols, dropna=False)[qty_col]
              .sum().rename("topk_sum").reset_index())

    share = (
        topk.merge(total.reset_index(), on=group_by_cols, how="left")
            .assign(**{out_col: lambda x: (x["topk_sum"] / x["group_total"]).fillna(0.0)})
            [group_by_cols + [out_col]]
    )
    return df.merge(share, on=group_by_cols, how="left")


# ===== 실행 예시 =====
# 파일 경로는 환경에 맞게 변경
df = pd.read_csv("train.csv")

# 날짜 변환
if "영업일자" in df.columns:
    df["영업일자"] = pd.to_datetime(df["영업일자"], errors="coerce")

# 업장명 보장 (영업장명 없으면 영업장명_메뉴명에서 추출)
df = ensure_store_col(df, full_col="영업장명_메뉴명", store_col="영업장명")

# 업장×일자 단위 쏠림도
group_cols = ["영업일자", "영업장명"]
df = add_top_menu_share(df, group_cols, top_k=1, out_col="top1_menu_share")
df = add_top_menu_share(df, group_cols, top_k=3, out_col="top3_menu_share")

# 저장
df.to_csv("train_with_top_share_by_store.csv", index=False)

# 확인용 출력(원하면 주석 해제)
# print(df[[*group_cols, "top1_menu_share", "top3_menu_share"]].drop_duplicates().head(20))
