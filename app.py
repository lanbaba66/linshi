import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


st.set_page_config(
    page_title="上海二手房价格分析",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@st.cache_data(ttl=3600)
def generate_house_data(n=2000):
    np.random.seed(2024)

    area = np.random.normal(130, 30, n).astype(int)
    area = np.clip(area, 60, 250)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.30, 0.40, 0.20, 0.05])
    bathrooms = np.random.choice([1, 2, 3], n, p=[0.30, 0.55, 0.15])
    age = np.random.exponential(10, n).astype(int)
    age = np.clip(age, 0, 40)
    distance_to_center = np.round(np.random.exponential(4, n) + 0.5, decimals=1)

    districts = np.random.choice(["浦东", "黄浦", "静安", "徐汇"], n, p=[0.30, 0.25, 0.20, 0.25])
    floors = np.random.choice(["低层", "中层", "高层"], n, p=[0.30, 0.40, 0.30])
    renovations = np.random.choice(["毛坯", "简装", "精装", "豪装"], n, p=[0.10, 0.30, 0.40, 0.20])

    districts_coef = {"浦东": 80, "黄浦": 40, "静安": 20, "徐汇": 60}
    floors_coef = {"低层": -15, "中层": 0, "高层": 20}
    reno_coef = {"毛坯": -30, "简装": -10, "精装": 10, "豪装": 40}

    district_factor = np.array([districts_coef[d] for d in districts])
    floor_factor = np.array([floors_coef[f] for f in floors])
    reno_factor = np.array([reno_coef[r] for r in renovations])

    base_price = (
        area * 1.5
        + bedrooms * 10
        + bathrooms * 8
        - age * 1.2
        - distance_to_center * 5
        + district_factor
        + floor_factor
        + reno_factor
    )
    noise = np.random.normal(loc=0, scale=0.12 * base_price)
    price = np.round(base_price + noise, decimals=0).astype(int)
    price = np.clip(price, 50, 800)

    return pd.DataFrame(
        {
            "price": price,
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age": age,
            "distance_to_center": distance_to_center,
            "floor": floors,
            "district": districts,
            "renovation": renovations,
        }
    )


def train_price_model(data):
    feature_cols = ["area", "bedrooms", "bathrooms", "age", "distance_to_center"]
    X = data[feature_cols]
    y = data["price"]
    if len(X) <= 10:
        return None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "model": model,
        "feature_cols": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }


def main():
    with st.spinner("正在加载模拟房源数据..."):
        df = generate_house_data()

    st.sidebar.title("🔎 筛选条件")
    selected_districts = st.sidebar.multiselect(
        "区域", options=sorted(df["district"].unique()), default=sorted(df["district"].unique())
    )
    selected_renovations = st.sidebar.multiselect(
        "装修情况", options=sorted(df["renovation"].unique()), default=sorted(df["renovation"].unique())
    )
    selected_floors = st.sidebar.multiselect(
        "楼层类型", options=sorted(df["floor"].unique()), default=sorted(df["floor"].unique())
    )

    price_min, price_max = int(df["price"].min()), int(df["price"].max())
    price_range = st.sidebar.slider(
        "总价范围（万元）", min_value=price_min, max_value=price_max, value=(price_min, price_max)
    )
    area_min, area_max = int(df["area"].min()), int(df["area"].max())
    area_range = st.sidebar.slider(
        "面积范围（㎡）", min_value=area_min, max_value=area_max, value=(area_min, area_max)
    )
    age_min, age_max = int(df["age"].min()), int(df["age"].max())
    age_range = st.sidebar.slider(
        "房龄范围（年）", min_value=age_min, max_value=age_max, value=(age_min, age_max)
    )

    filtered_df = df[
        (df["district"].isin(selected_districts))
        & (df["renovation"].isin(selected_renovations))
        & (df["floor"].isin(selected_floors))
        & (df["price"] >= price_range[0])
        & (df["price"] <= price_range[1])
        & (df["age"] >= age_range[0])
        & (df["age"] <= age_range[1])
        & (df["area"] >= area_range[0])
        & (df["area"] <= area_range[1])
    ]

    st.title("🏠 上海二手房价格分析与预测")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("房源数量", f"{len(filtered_df):,}")
    with col2:
        mean_price = filtered_df["price"].mean() if len(filtered_df) else 0
        st.metric("平均总价", f"{mean_price:.1f} 万元")
    with col3:
        median_price = filtered_df["price"].median() if len(filtered_df) else 0
        st.metric("中位总价", f"{median_price:.1f} 万元")
    with col4:
        if len(filtered_df):
            most_expensive = filtered_df.groupby("district")["price"].mean().idxmax()
        else:
            most_expensive = "-"
        st.metric("平均价最高区域", most_expensive)

    st.divider()

    if filtered_df.empty:
        st.warning("当前筛选条件下没有房源，请调整筛选范围。")
        return

    # 1. 房价分布（直方图 + 箱线图）
    st.subheader("1. 房价分布概览")
    fig_price, axes = plt.subplots(nrows=2, figsize=(12, 7))
    axes[0].hist(
        filtered_df["price"], bins=30, color="skyblue", edgecolor="black", alpha=0.7
    )
    axes[0].axvline(
        filtered_df["price"].mean(),
        color="red",
        linestyle="--",
        label=f"均值({filtered_df['price'].mean():.0f})",
    )
    axes[0].axvline(
        filtered_df["price"].median(),
        color="green",
        linestyle="--",
        label=f"中位数({filtered_df['price'].median():.0f})",
    )
    axes[0].set_title("房价分布直方图")
    axes[0].set_xlabel("总价（万元）")
    axes[0].set_ylabel("频数")
    axes[0].legend()

    axes[1].boxplot(
        filtered_df["price"],
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightgreen"),
    )
    axes[1].set_title("房价箱线图")
    axes[1].set_ylabel("总价（万元）")
    plt.tight_layout()
    st.pyplot(fig_price)

    # 2. 相关性热力图
    st.subheader("2. 特征相关性热力图")
    numeric_cols = ["price", "area", "bedrooms", "bathrooms", "age", "distance_to_center"]
    corr = filtered_df[numeric_cols].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    cax = ax_corr.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig_corr.colorbar(cax)
    ax_corr.set_xticks(range(len(corr.columns)))
    ax_corr.set_yticks(range(len(corr.columns)))
    ax_corr.set_xticklabels(corr.columns, rotation=30)
    ax_corr.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax_corr.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )
    ax_corr.set_title("特征相关性热力图", pad=20)
    st.pyplot(fig_corr)

    # 3. 面积 VS 价格
    st.subheader("3. 面积与房价关系")
    fig_area, ax_area = plt.subplots(figsize=(10, 5))
    ax_area.scatter(
        filtered_df["area"], filtered_df["price"], alpha=0.5, s=20, c="steelblue"
    )
    z = np.polyfit(filtered_df["area"], filtered_df["price"], deg=1)
    p = np.poly1d(z)
    x_line = np.linspace(filtered_df["area"].min(), filtered_df["area"].max(), num=100)
    ax_area.plot(
        x_line,
        p(x_line),
        color="red",
        linewidth=2,
        label=f"拟合: price={z[0]:.2f}*area+{z[1]:.2f}",
    )
    ax_area.set_xlabel("建筑面积（㎡）")
    ax_area.set_ylabel("总价（万元）")
    ax_area.legend()
    ax_area.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig_area)

    # 4. 各区域房价箱线图
    st.subheader("4. 各区域房价分布")
    districts_order = ["浦东", "黄浦", "静安", "徐汇"]
    if len(filtered_df) > 0:
        valid_districts = [
            d for d in districts_order if d in filtered_df["district"].unique()
        ]
        if valid_districts:
            data_to_plot = [
                filtered_df[filtered_df["district"] == d]["price"]
                for d in valid_districts
            ]
            fig_dist_box, ax_dist_box = plt.subplots(figsize=(10, 5))
            bp = ax_dist_box.boxplot(
                data_to_plot, tick_labels=valid_districts, patch_artist=True
            )
            colors = ["#ff9999", "#66b2ff", "#99ff99", "#ffcc99"]
            for patch, color in zip(bp["boxes"], colors[: len(data_to_plot)]):
                patch.set_facecolor(color)
            ax_dist_box.set_ylabel("总价（万元）")
            ax_dist_box.grid(axis="y", linestyle="--", alpha=0.5)
            st.pyplot(fig_dist_box)
        else:
            st.warning("当前筛选条件下没有有效区域数据，无法绘制区域房价图。")
    else:
        st.warning("当前筛选条件没有房源数据，请调整条件。")

    # 5. 装修情况 VS 房价
    st.subheader("5. 装修状态与房价")
    renovation_order = ["毛坯", "简装", "精装", "豪装"]
    renovation_states = (
        filtered_df.groupby("renovation")["price"]
        .agg(["mean", "std"])
        .reindex(renovation_order)
        .dropna()
    )
    if len(renovation_states):
        fig_reno, ax_reno = plt.subplots(figsize=(10, 5))
        bars = ax_reno.bar(
            renovation_states.index,
            renovation_states["mean"],
            yerr=renovation_states["std"],
            capsize=5,
            color=["#8bb6b2", "#cdbe70", "#ffd700", "#8b6508"][: len(renovation_states)],
            edgecolor="black",
        )
        ax_reno.set_ylabel("平均总价（万元）")
        ax_reno.grid(axis="y", linestyle="--", alpha=0.5)
        for bar in bars:
            height = bar.get_height()
            ax_reno.text(
                bar.get_x() + bar.get_width() / 2,
                height + 5,
                f"{height:.0f}",
                ha="center",
                va="bottom",
            )
        st.pyplot(fig_reno)
    else:
        st.warning("当前筛选条件下没有装修状态数据。")

    # 6. 房龄 VS 价格
    st.subheader("6. 房龄与房价关系")
    fig_age, ax_age = plt.subplots(figsize=(10, 5))
    ax_age.scatter(filtered_df["age"], filtered_df["price"], alpha=0.4, s=15, c="darkgreen")
    z2 = np.polyfit(filtered_df["age"], filtered_df["price"], deg=2)
    p2 = np.poly1d(z2)
    x_line2 = np.linspace(filtered_df["age"].min(), filtered_df["age"].max(), num=100)
    ax_age.plot(x_line2, p2(x_line2), color="red", linewidth=2, label="二次拟合")
    ax_age.set_xlabel("房龄（年）")
    ax_age.set_ylabel("总价（万元）")
    ax_age.legend()
    ax_age.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig_age)

    # 7. 距离市中心 VS 房价
    st.subheader("7. 距离市中心与房价")
    fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
    ax_dist.scatter(
        filtered_df["distance_to_center"],
        filtered_df["price"],
        alpha=0.4,
        s=15,
        c="purple",
    )
    z3 = np.polyfit(filtered_df["distance_to_center"], filtered_df["price"], deg=1)
    p3 = np.poly1d(z3)
    x_line3 = np.linspace(
        filtered_df["distance_to_center"].min(),
        filtered_df["distance_to_center"].max(),
        num=100,
    )
    ax_dist.plot(x_line3, p3(x_line3), color="red", linewidth=2, label="趋势线")
    ax_dist.set_xlabel("距市中心距离（km）")
    ax_dist.set_ylabel("总价（万元）")
    ax_dist.legend()
    ax_dist.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig_dist)

    # 8. 区域 × 装修交互分析
    st.subheader("8. 区域与装修情况交互分析")
    pivot = filtered_df.pivot_table(
        values="price",
        index="district",
        columns="renovation",
        aggfunc="mean",
        fill_value=0,
    )
    fig_pivot = px.imshow(
        pivot,
        text_auto=".0f",
        color_continuous_scale="YlOrRd",
        labels=dict(color="平均总价（万元）"),
        title="各区域 × 装修情况平均房价热力图",
    )
    st.plotly_chart(fig_pivot, use_container_width=True)

    st.divider()
    st.subheader("🤖 房价预测模型（线性回归）")
    result = train_price_model(filtered_df)
    if result is None:
        st.warning("筛选后房源过少，无法训练模型。")
    else:
        col1, col2 = st.columns(2)
        col1.metric("模型 MAE", f"{result['mae']:.2f} 万元")
        col2.metric("模型 R²", f"{result['r2']:.3f}")

        fig_pred, ax_pred = plt.subplots(figsize=(8, 6))
        ax_pred.scatter(result["y_test"], result["y_pred"], alpha=0.5, s=20)
        low = min(result["y_test"].min(), result["y_pred"].min())
        high = max(result["y_test"].max(), result["y_pred"].max())
        ax_pred.plot([low, high], [low, high], "r--", linewidth=2)
        ax_pred.set_xlabel("实际总价（万元）")
        ax_pred.set_ylabel("预测总价（万元）")
        ax_pred.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_pred)

        st.markdown("**特征系数**")
        coef_df = pd.DataFrame({"特征": result["feature_cols"], "系数": result["model"].coef_})
        st.dataframe(coef_df, use_container_width=True)

        st.markdown("**自定义房源预测**")
        col1, col2, col3 = st.columns(3)
        with col1:
            area_input = st.number_input("面积（㎡）", min_value=40, max_value=280, value=120)
            bedrooms_input = st.number_input("卧室数", min_value=1, max_value=5, value=3)
        with col2:
            bathrooms_input = st.number_input("卫生间数", min_value=1, max_value=3, value=2)
            age_input = st.number_input("房龄（年）", min_value=0, max_value=40, value=10)
        with col3:
            dist_input = st.number_input(
                "距市中心（km）", min_value=0.5, max_value=20.0, value=5.0, step=0.1
            )

        if st.button("预测房价"):
            input_data = np.array(
                [[area_input, bedrooms_input, bathrooms_input, age_input, dist_input]]
            )
            pred_price = result["model"].predict(input_data)[0]
            st.success(f"预测总价：{pred_price:.1f} 万元")

    st.divider()
    st.subheader("📋 数据明细")
    st.dataframe(
        filtered_df,
        use_container_width=True,
        column_config={
            "price": st.column_config.NumberColumn("总价（万元）", format="%.0f"),
            "area": "面积（㎡）",
            "bedrooms": "卧室",
            "bathrooms": "卫生间",
            "age": "房龄（年）",
            "distance_to_center": "距市中心（km）",
            "district": "区域",
            "floor": "楼层",
            "renovation": "装修情况",
        },
    )
    st.caption("💡 提示：可通过左侧筛选条件动态查看不同房源数据。")


if __name__ == "__main__":
    main()
