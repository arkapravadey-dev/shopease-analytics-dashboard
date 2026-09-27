# ShopEase dashboard - FBDA group project (group 011_023_048)
# Arkaprava Dey, Dhruv Singh, Tushar Raj
# run with:  streamlit run app.py

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

st.set_page_config(page_title="ShopEase Analytics | Group 011_023_048", layout="wide")
sns.set_theme(style="whitegrid", palette="deep")

GROUP_ID = "011_023_048"
SEED = 112348
SAMPLE_SIZE = 2500
ALPHA = 0.05
# live data link - raw GitHub link of shopease_raw_orders.csv
DATA_URL = "https://github.com/arkapravadey-dev/shopease-analytics-dashboard/blob/main/shopease_raw_orders.csv"
VALID_DISC = np.array([0.0, 0.05, 0.10, 0.15, 0.20])   # the only discount levels ShopEase uses


# ---- cleaning (same function as in the notebook) ----



def parse_date(x):
    # OrderDate comes in about six different formats, so we try each one in turn
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    if s.lower() in ("not available", "na", "n/a", ""):
        return pd.NaT
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%b %d %Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except ValueError:
            continue
    return pd.NaT


def clean_data(raw):
    df = raw.copy()
    notes = []            # we keep a short note of every fix so it can go into the report
    n_start = len(df)

    # text columns: extra spaces and mixed case ("male", " Male", "M", "FEMALE" ...)
    for c in ["Gender", "City", "Category", "Product", "PaymentMethod", "OrderStatus", "CustomerID"]:
        df[c] = df[c].str.strip()
    df["Gender"] = df["Gender"].str.lower().map({"male": "Male", "m": "Male", "female": "Female", "f": "Female"})
    df["City"] = df["City"].str.title()
    df["Category"] = df["Category"].str.title()
    df["Product"] = df["Product"].str.title()
    df["PaymentMethod"] = df["PaymentMethod"].str.title()
    df["OrderStatus"] = df["OrderStatus"].str.title().replace({"Canceled": "Cancelled"})
    notes.append("Trimmed spaces and fixed spelling/case in Gender, City, Category, Product, PaymentMethod and OrderStatus")

    # duplicates - first exact copies, then repeated OrderIDs (we keep the row with fewer blanks)
    exact_dups = df.duplicated().sum()
    df = df.drop_duplicates()
    df["_blanks"] = df.isna().sum(axis=1)
    df = df.sort_values(["OrderID", "_blanks"])
    id_dups = df["OrderID"].duplicated().sum()
    df = df.drop_duplicates("OrderID", keep="first").drop(columns="_blanks")
    notes.append(f"Removed {exact_dups} exact duplicate rows and {id_dups} more rows with a repeated OrderID")

    # dates
    df["OrderDate"] = df["OrderDate"].apply(parse_date)
    df["DeliveryDate"] = pd.to_datetime(df["DeliveryDate"], errors="coerce")
    notes.append(f"Converted OrderDate from its mixed formats; {df['OrderDate'].isna().sum()} dates marked 'not available' were left blank")

    # discount is stored both as 0.1 and as "10%"
    disc = pd.to_numeric(df["Discount"].str.strip().str.rstrip("%"), errors="coerce").astype(float)
    df["Discount"] = np.where(disc > 1, disc / 100, disc)

    # every product has one list price, so 0, -500 and 75k+ prices are entry errors
    list_price = df[df["UnitPrice"] > 0].groupby("Product")["UnitPrice"].agg(lambda s: s.mode().iloc[0])
    wrong_price = (df["UnitPrice"] != df["Product"].map(list_price)) & df["Product"].notna()
    df.loc[wrong_price, "UnitPrice"] = df.loc[wrong_price, "Product"].map(list_price)
    notes.append(f"Replaced {int(wrong_price.sum())} impossible UnitPrice values with the product's normal price")

    # a missing product name can be recovered when category + price point to only one product
    lookup = df.dropna(subset=["Product"]).groupby(["Category", "UnitPrice"])["Product"].agg(
        lambda s: s.mode().iloc[0] if s.nunique() == 1 else np.nan)
    no_product = df["Product"].isna()
    df.loc[no_product, "Product"] = [lookup.get((c, p), np.nan)
                                     for c, p in zip(df.loc[no_product, "Category"], df.loc[no_product, "UnitPrice"])]
    df["Product"] = df["Product"].fillna("Unknown")
    notes.append(f"Filled {int(no_product.sum() - (df['Product'] == 'Unknown').sum())} missing Product names using Category and UnitPrice")

    # negative quantity looks like a sign error; zero quantity is worked back from TotalAmount
    neg_qty = (df["Quantity"] < 0).sum()
    df["Quantity"] = df["Quantity"].abs()
    zero_qty = df["Quantity"] == 0
    d = df["Discount"].fillna(0)
    df.loc[zero_qty, "Quantity"] = (df.loc[zero_qty, "TotalAmount"] /
                                    (df.loc[zero_qty, "UnitPrice"] * (1 - d[zero_qty]))).round().clip(1, 4)
    df["Quantity"] = df["Quantity"].astype(int)
    notes.append(f"Fixed {neg_qty} negative quantities and {int(zero_qty.sum())} zero quantities")

    # missing discount: back it out of TotalAmount and round to the nearest real discount level
    no_disc = df["Discount"].isna()
    implied = 1 - df.loc[no_disc, "TotalAmount"] / (df.loc[no_disc, "Quantity"] * df.loc[no_disc, "UnitPrice"])
    df.loc[no_disc, "Discount"] = [VALID_DISC[np.abs(VALID_DISC - v).argmin()] for v in implied]
    notes.append(f"Filled {int(no_disc.sum())} missing discounts from TotalAmount / (Quantity x UnitPrice)")

    # recompute TotalAmount so every row adds up
    new_total = (df["Quantity"] * df["UnitPrice"] * (1 - df["Discount"])).round(2)
    changed = (~np.isclose(new_total, df["TotalAmount"])).sum()
    df["TotalAmount"] = new_total
    notes.append(f"Recalculated TotalAmount = Quantity x UnitPrice x (1 - Discount); {changed} rows changed")

    # ages like -3, 5 or 150 are not real customers -> blank, then median
    bad_age = (df["CustomerAge"] < 18) | (df["CustomerAge"] > 80)
    n_bad_age = bad_age.sum()
    df.loc[bad_age, "CustomerAge"] = np.nan
    n_missing_age = df["CustomerAge"].isna().sum()
    df["CustomerAge"] = df["CustomerAge"].fillna(df["CustomerAge"].median()).astype(int)
    notes.append(f"Blanked {n_bad_age} impossible ages and filled {n_missing_age} missing ages with the median")

    # rating must be 1-5; blanks are normal for orders that were never delivered
    bad_rating = (~df["Rating"].between(1, 5) & df["Rating"].notna()).sum()
    df.loc[~df["Rating"].between(1, 5), "Rating"] = np.nan
    notes.append(f"Removed {bad_rating} ratings outside 1-5 (blank ratings for undelivered orders were kept as blank)")

    df["CustomerID"] = df["CustomerID"].fillna("Unknown")
    top_payment = df["PaymentMethod"].mode().iloc[0]
    df["PaymentMethod"] = df["PaymentMethod"].fillna(top_payment)
    notes.append(f"Missing CustomerID set to 'Unknown'; missing PaymentMethod filled with the most common method ({top_payment})")

    # a few helper columns we use later
    df["DeliveryDays"] = (df["DeliveryDate"] - df["OrderDate"]).dt.days
    df.loc[df["DeliveryDays"] < 0, "DeliveryDays"] = np.nan
    df["OrderMonth"] = df["OrderDate"].dt.strftime("%Y-%m")
    df["DiscountPct"] = (df["Discount"] * 100).round().astype(int)
    df["AgeGroup"] = pd.cut(df["CustomerAge"], [17, 25, 35, 45, 55, 65, 80],
                            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66-80"]).astype(object)
    df["IsCancelled"] = (df["OrderStatus"] == "Cancelled").astype(int)
    df["HighRating"] = np.where(df["Rating"].isna(), np.nan, (df["Rating"] >= 4).astype(float))
    notes.append("Added DeliveryDays, OrderMonth, DiscountPct, AgeGroup, IsCancelled and HighRating")

    df = df.reset_index(drop=True)
    notes.append(f"Rows: {n_start} in the raw file -> {len(df)} after cleaning")
    return df, notes


# ---- load data once and keep it in cache ----
@st.cache_data(show_spinner="Loading and cleaning data...")
def load_all():
    raw = pd.read_csv(DATA_URL)
    clean, log = clean_data(raw)
    random.seed(SEED); np.random.seed(SEED)
    sample = clean.sample(n=SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)
    return raw, clean, sample, log


def show(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close("all")


def decision(p):
    return "Reject H₀ (significant)" if p < ALPHA else "Do not reject H₀ (not significant)"


def result_box(name, stat_name, stat, p, h0, extra=""):
    c1, c2, c3 = st.columns(3)
    c1.metric(stat_name, f"{stat:,.4f}")
    c2.metric("p-value", f"{p:.4g}")
    c3.metric("Decision (α = 0.05)", "Reject H₀" if p < ALPHA else "Do not reject H₀")
    st.caption(f"**{name}** — H₀: {h0}. {decision(p)}. {extra}")


try:
    raw, clean, sample, cleaning_log = load_all()
except Exception as e:
    st.error(f"Could not load data from DATA_URL: {e}")
    st.stop()

# ---- sidebar filters ----
st.sidebar.title("ShopEase Analytics")
st.sidebar.caption(f"Group **{GROUP_ID}** · seed **{SEED}**")
st.sidebar.caption(f"Random sample of {SAMPLE_SIZE:,} records")
base = sample

st.sidebar.markdown("### Filters")
dmin, dmax = base.OrderDate.min().date(), base.OrderDate.max().date()
date_range = st.sidebar.date_input("Order date range", (dmin, dmax), min_value=dmin, max_value=dmax)
f_city = st.sidebar.multiselect("City", sorted(base.City.unique()), default=sorted(base.City.unique()))
f_cat = st.sidebar.multiselect("Category", sorted(base.Category.unique()), default=sorted(base.Category.unique()))
f_gender = st.sidebar.multiselect("Gender", sorted(base.Gender.unique()), default=sorted(base.Gender.unique()))
f_pay = st.sidebar.multiselect("Payment method", sorted(base.PaymentMethod.unique()), default=sorted(base.PaymentMethod.unique()))
f_status = st.sidebar.multiselect("Order status", sorted(base.OrderStatus.unique()), default=sorted(base.OrderStatus.unique()))
f_age = st.sidebar.slider("Customer age", int(base.CustomerAge.min()), int(base.CustomerAge.max()),
                          (int(base.CustomerAge.min()), int(base.CustomerAge.max())))
f_disc = st.sidebar.multiselect("Discount %", sorted(base.DiscountPct.unique()), default=sorted(base.DiscountPct.unique()))

mask = (base.City.isin(f_city) & base.Category.isin(f_cat) & base.Gender.isin(f_gender) &
        base.PaymentMethod.isin(f_pay) & base.OrderStatus.isin(f_status) &
        base.CustomerAge.between(*f_age) & base.DiscountPct.isin(f_disc))
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    mask &= (base.OrderDate.dt.date.between(date_range[0], date_range[1]) | base.OrderDate.isna())
df = base[mask].copy()
rated = df.dropna(subset=["Rating"])
st.sidebar.success(f"{len(df):,} orders match the filters")

if len(df) < 30:
    st.warning("Fewer than 30 orders match the filters — widen the filters to run the analysis.")
    st.stop()

NUM_COLS = ["CustomerAge", "Quantity", "UnitPrice", "Discount", "TotalAmount", "Rating", "DeliveryDays"]
CAT_COLS = ["Gender", "City", "Category", "Product", "PaymentMethod", "OrderStatus", "AgeGroup", "DiscountPct"]

# ---- page header and headline numbers ----
st.title("ShopEase E-Commerce — Dynamic Analytical Dashboard")
st.caption(f"Foundations of Big Data Analytics with Python · Group {GROUP_ID} · random sample of {SAMPLE_SIZE:,} orders · all results react to the sidebar filters")

k = st.columns(6)
k[0].metric("Orders", f"{len(df):,}")
k[1].metric("Revenue (EGP)", f"{df.TotalAmount.sum():,.0f}")
k[2].metric("Avg order value", f"{df.TotalAmount.mean():,.0f}")
k[3].metric("Avg rating", f"{rated.Rating.mean():.2f} / 5" if len(rated) else "–")
k[4].metric("Cancellation rate", f"{df.IsCancelled.mean() * 100:.1f}%")
k[5].metric("Avg delivery days", f"{df.DeliveryDays.mean():.2f}")

tabs = st.tabs(["Description of Data", "Descriptive Statistics", "Visualization",
                "Inferential Statistics", "Regression"])

# ---- tab 1: description of data ----
with tabs[0]:
    st.subheader("Description of data")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Raw records", f"{len(raw):,}"); d2.metric("Clean records", f"{len(clean):,}")
    d3.metric("Variables (raw → clean)", f"{raw.shape[1]} → {clean.shape[1]}"); d4.metric("Sample", f"{SAMPLE_SIZE:,} (seed {SEED})")
    st.markdown("**Data type:** Cross-sectional (orders) with timestamps · **Span:** Static, Jan–Jun 2026 · **Source:** course dataset (`shopease_raw_orders.csv`)")
    st.subheader("Cleaning steps applied")
    for i, s in enumerate(cleaning_log, 1):
        st.markdown(f"{i}. {s}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Missing values — raw**")
        st.dataframe(raw.isna().sum().rename("Missing").to_frame().query("Missing > 0"))
    with c2:
        st.markdown("**Missing values — clean** (structural: not delivered / not rated)")
        st.dataframe(clean.isna().sum().rename("Missing").to_frame().query("Missing > 0"))
    view = st.radio("Preview", ["Filtered data", "Raw data"], horizontal=True)
    st.dataframe((df if view == "Filtered data" else raw).head(200), width="stretch")

# ---- tab 2: descriptive statistics ----
with tabs[1]:
    st.subheader("Non-categorical variables")
    rows = {}
    for c in NUM_COLS:
        x = df[c].dropna()
        if len(x) < 3: continue
        rows[c] = {"Count": x.count(), "Min": x.min(), "P25": x.quantile(.25), "Median": x.median(), "P75": x.quantile(.75),
                   "Max": x.max(), "Mean": x.mean(), "Mode": x.mode().iloc[0], "Range": x.max() - x.min(),
                   "Std Dev": x.std(), "Skewness": stats.skew(x), "Kurtosis": stats.kurtosis(x)}
    st.dataframe(pd.DataFrame(rows).T.style.format("{:,.3f}"), width="stretch")
    pct = st.slider("Custom percentile", 1, 99, 90)
    st.write({c: round(df[c].dropna().quantile(pct / 100), 3) for c in NUM_COLS})

    st.subheader("Correlation")
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
    cols = st.multiselect("Variables", NUM_COLS, default=NUM_COLS)
    if len(cols) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        sns.heatmap(df[cols].corr(method=method), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"{method.title()} correlation"); show(fig)

    st.subheader("Categorical variables")
    cv = st.selectbox("Variable", CAT_COLS + ["Rating"], index=2)
    vc = (rated if cv == "Rating" else df)[cv].value_counts()
    freq = pd.DataFrame({"Frequency": vc, "Relative frequency": vc / vc.sum(), "Percent": vc / vc.sum() * 100})
    c1, c2 = st.columns([1, 1.3])
    c1.dataframe(freq.style.format({"Relative frequency": "{:.4f}", "Percent": "{:.2f}%"}))
    c1.info(f"Highest: **{vc.idxmax()}** ({vc.max()}, {vc.max() / vc.sum() * 100:.1f}%) · "
            f"Lowest: **{vc.idxmin()}** ({vc.min()}, {vc.min() / vc.sum() * 100:.1f}%)")
    with c2:
        fig, ax = plt.subplots(figsize=(7, 4)); vc.plot.bar(ax=ax, color="steelblue"); ax.set_title(f"Frequency of {cv}"); show(fig)

# ---- tab 3: visualization ----
with tabs[2]:
    chart = st.selectbox("Chart type", ["Scatter plot", "Line plot (monthly)", "Box-whisker plot", "Violin plot",
                                        "Heat map (pivot)", "Pair plot", "Bar plot", "Histogram", "Pie chart"])
    cats = ["Category", "City", "Gender", "PaymentMethod", "OrderStatus", "AgeGroup", "DiscountPct"]
    fig = None
    if chart == "Scatter plot":
        c1, c2, c3, c4 = st.columns(4)
        x = c1.selectbox("X", NUM_COLS, index=0); y = c2.selectbox("Y", NUM_COLS, index=4)
        hue = c3.selectbox("Colour by", ["None"] + cats); logy = c4.checkbox("Log Y", value=(y == "TotalAmount"))
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df, x=x, y=y, hue=None if hue == "None" else hue, alpha=.5, ax=ax)
        if logy: ax.set_yscale("log")
        r, p = stats.pearsonr(df[[x, y]].dropna()[x], df[[x, y]].dropna()[y]); ax.set_title(f"{y} vs {x}  (Pearson r = {r:.3f}, p = {p:.3g})")
    elif chart == "Line plot (monthly)":
        c1, c2, c3 = st.columns(3)
        metric = c1.selectbox("Metric", ["TotalAmount", "OrderID", "Rating", "IsCancelled", "DeliveryDays"])
        agg = c2.selectbox("Aggregation", ["sum", "mean", "count"]); split = c3.selectbox("Split by", ["None"] + cats)
        d = df.dropna(subset=["OrderMonth"])
        fig, ax = plt.subplots(figsize=(10, 5))
        if split == "None":
            d.groupby("OrderMonth")[metric].agg(agg).plot(marker="o", ax=ax)
        else:
            d.pivot_table(index="OrderMonth", columns=split, values=metric, aggfunc=agg).plot(marker="o", ax=ax)
        ax.set_title(f"{agg}({metric}) by month")
    elif chart in ("Box-whisker plot", "Violin plot"):
        c1, c2, c3 = st.columns(3)
        y = c1.selectbox("Numeric variable", NUM_COLS, index=4); x = c2.selectbox("Group by", cats)
        logy = c3.checkbox("Log Y", value=(y == "TotalAmount"))
        fig, ax = plt.subplots(figsize=(10, 5))
        (sns.boxplot if chart.startswith("Box") else sns.violinplot)(data=df, x=x, y=y, ax=ax)
        if logy: ax.set_yscale("log")
        ax.set_title(f"{y} by {x}")
    elif chart == "Heat map (pivot)":
        c1, c2, c3, c4 = st.columns(4)
        r_ = c1.selectbox("Rows", cats, index=0); c_ = c2.selectbox("Columns", cats, index=1)
        v_ = c3.selectbox("Value", ["TotalAmount", "OrderID", "Rating", "IsCancelled", "Quantity"]); a_ = c4.selectbox("Agg", ["sum", "mean", "count"], index=1)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(df.pivot_table(index=r_, columns=c_, values=v_, aggfunc=a_), annot=True, fmt=".2f" if a_ == "mean" else ".0f", cmap="YlGnBu", ax=ax)
        ax.set_title(f"{a_}({v_}): {r_} × {c_}")
    elif chart == "Pair plot":
        cols = st.multiselect("Variables", NUM_COLS, default=["CustomerAge", "Quantity", "TotalAmount", "Rating"])
        hue = st.selectbox("Colour by", ["None"] + cats)
        if len(cols) >= 2:
            d = df[cols + ([] if hue == "None" else [hue])].dropna().sample(min(1500, len(df)), random_state=SEED)
            g = sns.pairplot(d, hue=None if hue == "None" else hue, corner=True, plot_kws={"alpha": .4, "s": 12}, height=2.2)
            show(g.figure)
    elif chart == "Bar plot":
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("Category axis", cats + ["Product"]); v = c2.selectbox("Value", ["Count", "TotalAmount", "Rating", "IsCancelled", "Quantity"])
        a = c3.selectbox("Aggregation", ["sum", "mean"])
        s = df[x].value_counts() if v == "Count" else df.groupby(x)[v].agg(a).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5)); s.plot.bar(ax=ax, color="teal"); ax.set_title(f"{v} by {x}")
    elif chart == "Histogram":
        c1, c2, c3 = st.columns(3)
        v = c1.selectbox("Variable", NUM_COLS, index=0); b = c2.slider("Bins", 5, 60, 20); lg = c3.checkbox("Log scale X")
        x = df[v].dropna(); x = np.log10(x) if lg else x
        fig, ax = plt.subplots(figsize=(10, 5)); sns.histplot(x, bins=b, kde=True, ax=ax); ax.set_title(f"Histogram of {'log10 ' if lg else ''}{v}")
    elif chart == "Pie chart":
        v = st.selectbox("Variable", cats)
        fig, ax = plt.subplots(figsize=(6, 6)); df[v].value_counts().plot.pie(autopct="%1.1f%%", startangle=90, ax=ax); ax.set_ylabel("")
    if fig is not None:
        show(fig)

# ---- tab 4: inferential statistics ----
with tabs[3]:
    test = st.selectbox("Choose analysis", [
        "Confidence interval", "Normality tests", "t-test (two groups)", "One-sample t-test", "ANOVA (k groups)",
        "Test of variance (F / Levene / Bartlett)", "Test of proportion (z-test)", "Correlation test (t)",
        "Chi-square goodness of fit", "Chi-square test of independence",
        "Mann-Whitney U", "Wilcoxon signed-rank", "Kruskal-Wallis H", "Friedman test"])
    group_cats = ["Gender", "Category", "City", "PaymentMethod", "OrderStatus", "AgeGroup", "DiscountPct"]

    def num_data(v):
        return rated if v == "Rating" else df

    if test == "Confidence interval":
        c1, c2 = st.columns(2)
        kind = c1.radio("Parameter", ["Mean", "Proportion"], horizontal=True)
        conf = c2.slider("Confidence level", .80, .99, .95, .01)
        if kind == "Mean":
            v = st.selectbox("Variable", NUM_COLS, index=4); x = num_data(v)[v].dropna()
            lo, hi = stats.t.interval(conf, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
            st.metric(f"Mean {v}", f"{x.mean():,.3f}", f"{conf:.0%} CI: [{lo:,.3f}, {hi:,.3f}]", delta_color="off")
            by = st.selectbox("CI by group", ["None"] + group_cats)
            if by != "None":
                rows = []
                for g, d in num_data(v).groupby(by):
                    y = d[v].dropna()
                    if len(y) > 2:
                        l, h = stats.t.interval(conf, len(y) - 1, loc=y.mean(), scale=stats.sem(y)); rows.append([g, y.mean(), l, h])
                t = pd.DataFrame(rows, columns=[by, "Mean", "Lower", "Upper"])
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.errorbar(t[by].astype(str), t.Mean, yerr=[t.Mean - t.Lower, t.Upper - t.Mean], fmt="o", capsize=6)
                ax.set_title(f"Mean {v} with {conf:.0%} CI by {by}"); show(fig); st.dataframe(t)
        else:
            ev = st.selectbox("Event", ["Order cancelled", "High rating (4-5)", "Order delivered"])
            y = {"Order cancelled": df.IsCancelled, "High rating (4-5)": rated.HighRating, "Order delivered": (df.OrderStatus == "Delivered").astype(int)}[ev]
            lo, hi = proportion_confint(y.sum(), len(y), alpha=1 - conf, method="wilson")
            st.metric(f"P({ev})", f"{y.mean():.2%}", f"{conf:.0%} CI: [{lo:.2%}, {hi:.2%}]", delta_color="off")

    elif test == "Normality tests":
        v = st.selectbox("Variable", NUM_COLS, index=4); lg = st.checkbox("Log-transform", value=(v == "TotalAmount"))
        x = num_data(v)[v].dropna(); x = np.log(x) if lg else x
        sw = stats.shapiro(x.sample(min(5000, len(x)), random_state=SEED)); ks = stats.kstest((x - x.mean()) / x.std(), "norm")
        ad = stats.anderson(x, dist="norm"); jb = stats.jarque_bera(x)
        st.dataframe(pd.DataFrame({"Test": ["Shapiro-Wilk", "Kolmogorov-Smirnov", "Anderson-Darling", "Jarque-Bera"],
                                   "Statistic": [sw.statistic, ks.statistic, ad.statistic, jb.statistic],
                                   "p-value / 5% critical": [f"{sw.pvalue:.4g}", f"{ks.pvalue:.4g}", f"crit = {ad.critical_values[2]:.3f}", f"{jb.pvalue:.4g}"],
                                   "Normal at 5%?": ["Yes" if sw.pvalue > ALPHA else "No", "Yes" if ks.pvalue > ALPHA else "No",
                                                     "Yes" if ad.statistic < ad.critical_values[2] else "No", "Yes" if jb.pvalue > ALPHA else "No"]}))
        fig, ax = plt.subplots(figsize=(10, 4)); sns.histplot(x, kde=True, ax=ax); ax.set_title("Histogram"); show(fig)

    elif test in ("t-test (two groups)", "Mann-Whitney U"):
        c1, c2 = st.columns(2)
        v = c1.selectbox("Numeric variable", NUM_COLS, index=4); g = c2.selectbox("Group variable", group_cats)
        d = num_data(v).dropna(subset=[v]); levels = sorted(d[g].unique(), key=str)
        c3, c4 = st.columns(2)
        a = c3.selectbox("Group A", levels, index=0); b = c4.selectbox("Group B", levels, index=min(1, len(levels) - 1))
        xa, xb = d.loc[d[g] == a, v], d.loc[d[g] == b, v]
        st.write(f"Group A ({a}): n = {len(xa)}, mean = {xa.mean():,.3f}, median = {xa.median():,.3f} · "
                 f"Group B ({b}): n = {len(xb)}, mean = {xb.mean():,.3f}, median = {xb.median():,.3f}")
        if test.startswith("t-test"):
            eq = st.checkbox("Assume equal variances (Student's t)", value=False)
            r = stats.ttest_ind(xa, xb, equal_var=eq)
            result_box("Student's t-test" if eq else "Welch's t-test", "t statistic", r.statistic, r.pvalue, f"mean {v} of {a} = {b}")
        else:
            r = stats.mannwhitneyu(xa, xb)
            result_box("Mann-Whitney U", "U statistic", r.statistic, r.pvalue, f"{v} distributions of {a} and {b} are the same")

    elif test == "One-sample t-test":
        v = st.selectbox("Variable", NUM_COLS, index=5); x = num_data(v)[v].dropna()
        mu = st.number_input("Hypothesised mean μ₀", value=float(round(x.mean(), 0)))
        r = stats.ttest_1samp(x, mu)
        result_box("One-sample t-test", "t statistic", r.statistic, r.pvalue, f"mean {v} = {mu}", f"Sample mean = {x.mean():.4f}")

    elif test in ("ANOVA (k groups)", "Kruskal-Wallis H"):
        c1, c2 = st.columns(2)
        v = c1.selectbox("Numeric variable", NUM_COLS, index=4); g = c2.selectbox("Group variable", group_cats, index=1)
        d = num_data(v).dropna(subset=[v]); groups = [x[v].values for _, x in d.groupby(g) if len(x) > 1]
        if test.startswith("ANOVA"):
            r = stats.f_oneway(*groups); result_box("One-way ANOVA", "F statistic", r.statistic, r.pvalue, f"mean {v} equal across {g}")
        else:
            r = stats.kruskal(*groups); result_box("Kruskal-Wallis H", "H statistic", r.statistic, r.pvalue, f"{v} distribution equal across {g}")
        st.dataframe(d.groupby(g)[v].agg(["count", "mean", "median", "std"]).style.format("{:,.3f}"))
        fig, ax = plt.subplots(figsize=(10, 4)); sns.boxplot(data=d, x=g, y=v, ax=ax)
        if v == "TotalAmount": ax.set_yscale("log")
        show(fig)

    elif test == "Test of variance (F / Levene / Bartlett)":
        c1, c2 = st.columns(2)
        v = c1.selectbox("Numeric variable", NUM_COLS, index=4); g = c2.selectbox("Group variable", group_cats, index=1)
        lg = st.checkbox("Log-transform", value=(v == "TotalAmount"))
        d = num_data(v).dropna(subset=[v]).copy()
        if lg: d[v] = np.log(d[v])
        groups = [x[v].values for _, x in d.groupby(g) if len(x) > 1]
        lev = stats.levene(*groups); bar = stats.bartlett(*groups)
        result_box("Levene test", "W statistic", lev.statistic, lev.pvalue, "variances equal across groups")
        result_box("Bartlett test", "χ² statistic", bar.statistic, bar.pvalue, "variances equal across groups (assumes normality)")
        if len(groups) == 2:
            F = np.var(groups[0], ddof=1) / np.var(groups[1], ddof=1); d1, d2 = len(groups[0]) - 1, len(groups[1]) - 1
            p = 2 * min(stats.f.cdf(F, d1, d2), 1 - stats.f.cdf(F, d1, d2))
            result_box("F-test (two variances)", "F statistic", F, p, "σ²₁ = σ²₂")
        else:
            st.caption("The two-sample F-test appears when the group variable has exactly two levels (e.g. Gender).")
        st.dataframe(d.groupby(g)[v].agg(["count", "var", "std"]).style.format("{:,.4f}"))

    elif test == "Test of proportion (z-test)":
        ev = st.selectbox("Event", ["Order cancelled", "High rating (4-5)"])
        d = df if ev == "Order cancelled" else rated; col = "IsCancelled" if ev == "Order cancelled" else "HighRating"
        mode = st.radio("Test", ["One proportion vs benchmark", "Two groups"], horizontal=True)
        if mode.startswith("One"):
            p0 = st.slider("Benchmark proportion p₀", .01, .99, .10 if col == "IsCancelled" else .75, .01)
            z, p = proportions_ztest(d[col].sum(), len(d), value=p0)
            result_box("One-proportion z-test", "z statistic", z, p, f"P({ev}) = {p0}", f"Observed = {d[col].mean():.3%}")
        else:
            g = st.selectbox("Group variable", group_cats); lv = sorted(d[g].unique(), key=str)
            c1, c2 = st.columns(2); a = c1.selectbox("Group A", lv, 0); b = c2.selectbox("Group B", lv, min(1, len(lv) - 1))
            cnt = [d.loc[d[g] == a, col].sum(), d.loc[d[g] == b, col].sum()]; nob = [(d[g] == a).sum(), (d[g] == b).sum()]
            z, p = proportions_ztest(cnt, nob)
            result_box("Two-proportion z-test", "z statistic", z, p, f"P({ev}) equal for {a} and {b}",
                       f"{a}: {cnt[0] / nob[0]:.2%} · {b}: {cnt[1] / nob[1]:.2%}")

    elif test == "Correlation test (t)":
        c1, c2 = st.columns(2)
        x = c1.selectbox("X", NUM_COLS, index=6); y = c2.selectbox("Y", NUM_COLS, index=5)
        d = df[[x, y]].dropna(); r = d[x].corr(d[y]); n = len(d)
        t = r * np.sqrt((n - 2) / (1 - r ** 2)) if abs(r) < 1 else np.inf; p = 2 * (1 - stats.t.cdf(abs(t), n - 2))
        result_box("t-test for Pearson correlation", "t statistic", t, p, "ρ = 0", f"r = {r:.4f}, n = {n}")
        sp = stats.spearmanr(d[x], d[y]); st.caption(f"Spearman ρ = {sp.statistic:.4f}, p = {sp.pvalue:.4g}")

    elif test == "Chi-square goodness of fit":
        v = st.selectbox("Categorical variable", ["Category", "PaymentMethod", "City", "Gender", "OrderStatus", "DiscountPct"])
        obs = df[v].value_counts().sort_index()
        st.caption("H₀: all categories are equally likely (uniform expected frequencies)")
        r = stats.chisquare(obs)
        result_box("Chi-square goodness of fit", "χ² statistic", r.statistic, r.pvalue, f"{v} is uniformly distributed")
        st.dataframe(pd.DataFrame({"Observed": obs, "Expected": obs.sum() / len(obs)}))

    elif test == "Chi-square test of independence":
        c1, c2 = st.columns(2)
        a = c1.selectbox("Variable 1", group_cats, index=1); b = c2.selectbox("Variable 2", ["OrderStatus", "PaymentMethod", "Gender", "City", "Category", "AgeGroup", "Rating"])
        d = rated if b == "Rating" else df; tab = pd.crosstab(d[a], d[b])
        r = stats.chi2_contingency(tab)
        result_box("Chi-square test of independence", "χ² statistic", r.statistic, r.pvalue, f"{a} and {b} are independent", f"df = {r.dof}")
        fig, ax = plt.subplots(figsize=(10, 4)); sns.heatmap(pd.crosstab(d[a], d[b], normalize="index") * 100, annot=True, fmt=".1f", cmap="Blues", ax=ax)
        ax.set_title(f"Row % of {b} within {a}"); show(fig)

    elif test == "Wilcoxon signed-rank":
        mode = st.radio("Design", ["One sample vs median", "Paired: gross vs net amount (discounted orders)"])
        if mode.startswith("One"):
            v = st.selectbox("Variable", NUM_COLS, index=5); x = num_data(v)[v].dropna()
            m0 = st.number_input("Hypothesised median", value=float(x.median()))
            diff = x - m0; diff = diff[diff != 0]
            r = stats.wilcoxon(diff) if len(diff) > 10 else None
            if r: result_box("Wilcoxon signed-rank", "W statistic", r.statistic, r.pvalue, f"median {v} = {m0}")
            else: st.info("All values equal the hypothesised median — choose another value.")
        else:
            d = df[df.Discount > 0]; r = stats.wilcoxon(d.Quantity * d.UnitPrice, d.TotalAmount)
            result_box("Wilcoxon signed-rank (paired)", "W statistic", r.statistic, r.pvalue, "gross and net amounts are equal")

    elif test == "Friedman test":
        v = st.selectbox("Measure per month", ["TotalAmount (sum)", "Orders (count)", "Rating (mean)"])
        treat = st.selectbox("Treatments (compared)", ["Category", "PaymentMethod", "City"])
        col, agg = {"TotalAmount (sum)": ("TotalAmount", "sum"), "Orders (count)": ("OrderID", "count"), "Rating (mean)": ("Rating", "mean")}[v]
        tab = df.dropna(subset=["OrderMonth"]).pivot_table(index="OrderMonth", columns=treat, values=col, aggfunc=agg).dropna()
        st.caption("Blocks = months (repeated measures); treatments = levels of the chosen variable")
        if tab.shape[0] >= 2 and tab.shape[1] >= 3:
            r = stats.friedmanchisquare(*[tab[c] for c in tab.columns])
            result_box("Friedman test", "χ² statistic", r.statistic, r.pvalue, f"{v} identical across {treat}")
            st.dataframe(tab.style.format("{:,.2f}"))
        else:
            st.info("Need at least 3 treatment levels and 2 months after filtering.")

# ---- tab 5: regression ----
with tabs[4]:
    model = st.selectbox("Model", ["Multiple linear regression (cross-sectional)", "Time-series regression (daily trend)",
                                    "Polynomial regression", "Regression with categorical variables", "Logistic regression"])
    if model.startswith("Multiple"):
        y = st.selectbox("Dependent variable", ["log(TotalAmount)", "TotalAmount", "Rating"])
        preds = st.multiselect("Predictors", ["Quantity", "UnitPrice", "np.log(UnitPrice)", "Discount", "CustomerAge", "DeliveryDays"],
                               default=["Quantity", "np.log(UnitPrice)", "Discount", "CustomerAge"])
        if preds:
            yv = "np.log(TotalAmount)" if y.startswith("log") else y
            d = rated if y == "Rating" or "DeliveryDays" in preds else df
            d = d.dropna(subset=[c for c in ["DeliveryDays"] if c in preds])
            m = smf.ols(f"{yv} ~ " + " + ".join(preds), data=d).fit()
            c1, c2, c3 = st.columns(3)
            c1.metric("R²", f"{m.rsquared:.4f}"); c2.metric("Adj. R²", f"{m.rsquared_adj:.4f}"); c3.metric("F p-value", f"{m.f_pvalue:.4g}")
            st.dataframe(pd.DataFrame({"Coefficient": m.params, "Std error": m.bse, "t": m.tvalues, "p-value": m.pvalues}).style.format("{:.4f}"))

    elif model.startswith("Time"):
        target = st.radio("Daily series", ["Revenue", "Orders"], horizontal=True)
        daily = df.dropna(subset=["OrderDate"]).set_index("OrderDate").resample("D").agg(Revenue=("TotalAmount", "sum"), Orders=("OrderID", "count")).reset_index()
        daily["t"] = np.arange(len(daily))
        m = smf.ols(f"{target} ~ t", data=daily).fit()
        c1, c2, c3 = st.columns(3)
        c1.metric("Trend per day", f"{m.params['t']:,.3f}"); c2.metric("p-value (trend)", f"{m.pvalues['t']:.4g}"); c3.metric("R²", f"{m.rsquared:.4f}")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(daily.OrderDate, daily[target], alpha=.4, label=f"Daily {target}")
        ax.plot(daily.OrderDate, daily[target].rolling(7).mean(), lw=2, label="7-day moving average")
        ax.plot(daily.OrderDate, m.fittedvalues, "r--", lw=2, label="Linear trend"); ax.legend(); show(fig)

    elif model.startswith("Polynomial"):
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X", ["DeliveryDays", "CustomerAge", "Discount", "Quantity"]); y = c2.selectbox("Y", ["Rating", "log(TotalAmount)", "TotalAmount"])
        deg = c3.slider("Degree", 1, 4, 2)
        d = (rated if y == "Rating" else df).dropna(subset=[x]).copy(); d["_y"] = np.log(d.TotalAmount) if y.startswith("log") else d[y]
        terms = " + ".join([x] + [f"I({x}**{k})" for k in range(2, deg + 1)])
        m = smf.ols(f"_y ~ {terms}", data=d).fit()
        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{m.rsquared:.4f}"); c2.metric("Adj. R²", f"{m.rsquared_adj:.4f}"); c3.metric("F p-value", f"{m.f_pvalue:.4g}")
        st.dataframe(pd.DataFrame({"Coefficient": m.params, "p-value": m.pvalues}).style.format("{:.5f}"))
        grid = pd.DataFrame({x: np.linspace(d[x].min(), d[x].max(), 100)})
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(d[x], d["_y"], alpha=.15, s=10, label="Observed")
        mean_y = d.groupby(x)["_y"].mean()
        if len(mean_y) <= 60: ax.scatter(mean_y.index, mean_y.values, color="k", s=40, label="Group mean")
        ax.plot(grid[x], m.predict(grid), "r-", lw=2, label=f"Degree {deg} fit"); ax.set_xlabel(x); ax.set_ylabel(y); ax.legend(); show(fig)

    elif model.startswith("Regression with categorical"):
        y = st.selectbox("Dependent variable", ["log(TotalAmount)", "Rating"])
        cats_sel = st.multiselect("Categorical predictors", ["Category", "City", "Gender", "PaymentMethod", "AgeGroup"], default=["Category", "City", "Gender", "PaymentMethod"])
        nums_sel = st.multiselect("Numeric controls", ["Quantity", "Discount", "CustomerAge", "DeliveryDays"], default=["Quantity", "Discount"])
        if cats_sel or nums_sel:
            yv = "np.log(TotalAmount)" if y.startswith("log") else "Rating"
            d = rated if y == "Rating" else df
            if "DeliveryDays" in nums_sel: d = d.dropna(subset=["DeliveryDays"])
            m = smf.ols(f"{yv} ~ " + " + ".join([f"C({c})" for c in cats_sel] + nums_sel), data=d).fit()
            c1, c2, c3 = st.columns(3)
            c1.metric("R²", f"{m.rsquared:.4f}"); c2.metric("Adj. R²", f"{m.rsquared_adj:.4f}"); c3.metric("F p-value", f"{m.f_pvalue:.4g}")
            st.markdown("**ANOVA table (Type II) — which factors matter?**")
            st.dataframe(sm.stats.anova_lm(m, typ=2).style.format("{:.4f}"))
            coefs = pd.DataFrame({"Coefficient": m.params, "p-value": m.pvalues}).drop("Intercept")
            fig, ax = plt.subplots(figsize=(10, max(3, len(coefs) * .3)))
            ax.barh(coefs.index, coefs.Coefficient, color=np.where(coefs["p-value"] < ALPHA, "seagreen", "lightgray"))
            ax.axvline(0, color="k"); ax.set_title("Coefficients (green = significant at 5%)"); show(fig)

    else:
        target = st.radio("Outcome (1 = event)", ["High rating (4-5)", "Order cancelled"], horizontal=True)
        preds = st.multiselect("Predictors", ["Discount", "CustomerAge", "DeliveryDays", "np.log(TotalAmount)", "C(Category)", "C(Gender)", "C(PaymentMethod)", "C(City)"],
                               default=["Discount", "CustomerAge", "C(Category)", "C(PaymentMethod)"])
        if target.startswith("Order") and "DeliveryDays" in preds:
            st.warning("DeliveryDays is only known for delivered orders, so it is dropped for the cancellation model."); preds = [p for p in preds if p != "DeliveryDays"]
        if preds:
            yv = "HighRating" if target.startswith("High") else "IsCancelled"
            d = (rated if yv == "HighRating" else df).dropna(subset=[p for p in ["DeliveryDays"] if p in preds])
            try:
                m = smf.logit(f"{yv} ~ " + " + ".join(preds), data=d).fit(disp=0)
                c1, c2, c3 = st.columns(3)
                c1.metric("McFadden pseudo-R²", f"{m.prsquared:.4f}"); c2.metric("LLR p-value", f"{m.llr_pvalue:.4g}"); c3.metric("Base rate", f"{d[yv].mean():.2%}")
                st.dataframe(pd.DataFrame({"Coefficient": m.params, "Odds ratio": np.exp(m.params), "p-value": m.pvalues}).style.format("{:.4f}"))
            except Exception as e:
                st.error(f"Model could not be estimated with these filters/predictors: {e}")
