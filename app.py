"""Group 9 — Netflix customer churn classroom application."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Churn Studio | Group 9", page_icon="🎬", layout="wide")

@st.cache_resource
def load_models():
    return (joblib.load(ROOT / "random_forest_pipeline.joblib"),
            joblib.load(ROOT / "logistic_regression_pipeline.joblib"),
            joblib.load(ROOT / "clustering_bundle.joblib"))

@st.cache_data
def table(name):
    return pd.read_csv(ROOT / name, index_col=0)

@st.cache_data
def aggregates():
    return pd.read_csv(ROOT / "dashboard_aggregates.csv", keep_default_na=False)

try:
    rf, lr, cluster = load_models()
    data = aggregates()
except Exception as exc:
    st.error(f"The application could not load its assets: {exc}")
    st.info("Keep all supplied files together and install requirements.txt. See README.md.")
    st.stop()

st.markdown('''<style>
.stApp {background:#101219;color:#f1f3f8;}
[data-testid="stSidebar"] {background:#191d28;}
[data-testid="stMetric"] {background:#202635;border:1px solid #343e54;padding:18px;border-radius:12px;}
h1,h2,h3 {letter-spacing:-.03em;}
.block-container {padding-top:2rem;max-width:1450px;}
</style>''', unsafe_allow_html=True)
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = ["#ed4264", "#4fd1c5", "#a6a9ff", "#ffca80"]

st.sidebar.markdown("## 🎬 CHURN STUDIO\n**GROUP 9 · MBA ANALYTICS**")
page = st.sidebar.radio("Navigate", ["Dashboard", "Explore Data", "Customer Segments", "Churn Prediction", "Model Comparison", "Business Insights"])
st.sidebar.caption("Educational project using an unverified-source dataset. Not affiliated with Netflix.")
st.title(page)
st.caption("NETFLIX CUSTOMER CHURN · Evidence → interpretation → retention decisions")


def chart(fig):
    fig.update_layout(margin=dict(l=10,r=10,t=45,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


def cohort_view():
    a,b,c=st.columns(3)
    scope=a.selectbox("Dataset", ["Training (4,000)", "Full sample (5,000)"])
    plan=b.selectbox("Subscription plan", ["All"]+sorted(data.plan.unique().tolist()))
    region=c.selectbox("Region", ["All"]+sorted(data.region.unique().tolist()))
    d=data[data.scope.eq("training" if scope.startswith("Training") else "full")]
    if plan!="All": d=d[d.plan.eq(plan)]
    if region!="All": d=d[d.region.eq(region)]
    return d


if page in ["Dashboard", "Explore Data"]:
    d=cohort_view()
    base=d[d.category.eq("All")]
    n=int(base.customers.sum()); churn=int(base.churned.sum())
    if n==0:
        st.info("No observations match these filters."); st.stop()
    if page=="Dashboard":
        cols=st.columns(4)
        for col,label,value in zip(cols,["Customer records","Recorded churn","Not flagged churned","Mean viewing hours"],
                                  [f"{n:,}",f"{churn/n:.1%}",f"{1-churn/n:.1%}",f"{base.watch_sum.sum()/n:.2f}"]):
            col.metric(label,value)
        st.caption("Churn is the proportion labelled 1 in this sample; it is not a verified monthly or annual rate. Viewing-hours measurement period and fee currency are unspecified.")
        by_plan=base.groupby("plan",as_index=False)[["customers","churned"]].sum()
        by_plan["Recorded churn (%)"]=100*by_plan.churned/by_plan.customers
        l,r=st.columns(2)
        with l: chart(px.bar(by_plan,x="plan",y="customers",title="Customer distribution"))
        with r: chart(px.bar(by_plan,x="plan",y="Recorded churn (%)",range_y=[0,100],title="Recorded churn within plan"))
        st.info("Managerial question: Where should limited retention attention be focused—broad campaigns, customer segments, or individual scores?")
    else:
        labels={"gender":"Gender","device":"Device","payment_method":"Payment method","favorite_genre":"Favourite genre","number_of_profiles":"Profile count"}
        feature=st.selectbox("Compare customer groups",list(labels),format_func=labels.get)
        summary=d[d.category.eq(feature)].groupby("value",as_index=False)[["customers","churned"]].sum()
        summary["Recorded churn (%)"]=100*summary.churned/summary.customers
        summary=summary.sort_values("Recorded churn (%)",ascending=False)
        fig=px.bar(summary,x="value",y="Recorded churn (%)",hover_data=["customers","churned"],range_y=[0,100],title=labels[feature])
        fig.add_hline(y=100*churn/n,line_dash="dash",annotation_text="Selected cohort baseline")
        chart(fig)
        st.dataframe(summary.rename(columns={"value":"Group"}),hide_index=True,use_container_width=True)
        st.download_button("Download this group summary",summary.to_csv(index=False),"group_summary.csv","text/csv")
        st.caption("Descriptive associations do not establish causes. Small differences can reflect sampling variation; these charts do not test statistical significance.")

elif page=="Customer Segments":
    profiles=table("cluster_profiles.csv")
    outcomes=table("cluster_churn.csv")
    combined=profiles.join(outcomes.drop(columns="Customers"))
    descriptions={0:"Recent login · more profiles · modest viewing",1:"Long login gap · fewer profiles · mostly Basic",2:"Long login gap · higher viewing · mainly Premium",3:"Recent login · Basic plan",4:"Long login gap · very low viewing",5:"Recent login · higher viewing · fewer profiles",6:"Long login gap · modest viewing · mostly Basic"}
    st.info("Seven exploratory groups fitted to 4,000 training customers. Full-training silhouette: 0.2255; groups overlap. Cluster numbers are identifiers, not ranks.")
    chart(px.bar(combined.reset_index(),x="Cluster",y="Recorded_Churn_Percentage",hover_data=["Customers"],range_y=[0,100],title="Recorded churn across training clusters"))
    selected=st.selectbox("Inspect a segment",list(combined.index),format_func=lambda v:f"Cluster {v} — {descriptions[v]}")
    row=combined.loc[selected]
    cols=st.columns(4)
    for c,label,val in zip(cols,["Customers","Median viewing hours","Median login gap (days)","Recorded churn"],[f"{row.Customers:,.0f}",f"{row.Median_Watch_Hours:.2f}",f"{row.Median_Login_Days:.0f}",f"{row.Recorded_Churn_Percentage:.2f}%"]): c.metric(label,val)
    st.dataframe(table("cluster_subscription_mix.csv").loc[[selected]],use_container_width=True)
    st.caption("Plan composition is percentage within cluster. Monthly fees in the profile table have unspecified currency.")
    st.dataframe(combined.round(2),use_container_width=True)
    st.markdown("**Interpretation:** clusters 1, 4 and 6 merit investigation because their recorded churn exceeds the 50.3% training baseline. Eligibility and pre-churn timing must be verified before contacting customers.")

elif page=="Churn Prediction":
    st.info("Educational model estimate of the recorded churn label. There is no verified prediction horizon or probability calibration. Use hypothetical profiles; do not enter personal identifiers.")
    schema=json.loads((ROOT/"input_schema.json").read_text())
    values={}
    with st.form("customer"):
        left,right=st.columns(2)
        for i,(name,label) in enumerate([("age","Age"),("watch_hours","Recorded viewing hours"),("last_login_days","Days since last login"),("number_of_profiles","Number of profiles")]):
            spec=schema["numeric"][name]
            with left:
                values[name]=st.number_input(label,min_value=spec["min"],max_value=spec["max"],value=spec["default"],step=spec["step"])
        for name,options in schema["categorical"].items():
            with right: values[name]=st.selectbox(name.replace("_"," ").title(),options)
        submitted=st.form_submit_button("Estimate recorded churn",use_container_width=True)
    st.caption("Inputs are limited to observed training ranges. Viewing hours must use the dataset's recorded measurement basis; that period has not been independently verified.")
    if submitted:
        features=list(rf.feature_names_in_)
        x=pd.DataFrame([values],columns=features)
        if not np.isfinite(x[schema["numeric"].keys()].to_numpy(dtype=float)).all():
            st.error("Enter finite numbers for every numerical input."); st.stop()
        p=float(rf.predict_proba(x)[0,list(rf.classes_).index(1)])
        pred=int(rf.predict(x)[0])
        st.metric("Random Forest model score for churn label 1",f"{p:.1%}")
        st.write("**Predicted label:** "+("Churned (1)" if pred==1 else "Not churned (0)"))
        st.caption("The saved classifier's standard decision rule is used. This score is not a confirmed chance of future churn or evidence that an intervention will work.")
        fees={"Basic":8.99,"Standard":13.99,"Premium":17.99}
        raw=pd.DataFrame([{**values,"monthly_fee":fees[values["subscription_type"]]}])[cluster["raw_feature_order"]].copy()
        raw["watch_hours"]=np.log1p(raw["watch_hours"])
        raw=raw.rename(columns={"watch_hours":"log_watch_hours"})[cluster["scaled_feature_order"]]
        segment=int(cluster["model"].predict(cluster["scaler"].transform(raw))[0])
        st.write(f"**Nearest exploratory segment:** Cluster {segment}")
        st.caption("Segment assignment uses the original fitted scaler and K-Means model. The monthly fee is derived from the selected plan using the dataset's fixed plan–fee mapping.")
        result=x.copy(); result["model_score_label_1"]=p;result["predicted_label"]=pred;result["Cluster"]=segment
        st.download_button("Download this hypothetical estimate",result.to_csv(index=False),"hypothetical_prediction.csv","text/csv")

elif page=="Model Comparison":
    st.subheader("Held-out test · 1,000 customers")
    results=table("test_model_comparison.csv")
    st.dataframe(results.style.format({"Accuracy":"{:.2%}","Precision":"{:.2%}","Recall":"{:.2%}","F1":"{:.4f}"}),use_container_width=True)
    plot=results[["Accuracy","Precision","Recall","F1"]].reset_index().melt(id_vars="Model",var_name="Metric",value_name="Score")
    chart(px.bar(plot,x="Metric",y="Score",color="Model",barmode="group",range_y=[0,1],title="Test performance at the saved decision rule"))
    st.write("**Random Forest confusion matrix:** 482 true negatives, 15 false positives, 19 false negatives, 484 true positives.")
    st.caption("Random Forest was selected using training cross-validation before test evaluation. Test results are fixed and are not recomputed from dashboard filters.")
    with st.expander("Training cross-validation and methodology"):
        st.dataframe(table("cross_validation_comparison.csv"),use_container_width=True)
        st.write("Stratified 80/20 split, random state 42. Three-fold stratified CV; hyperparameter search refit on F1. Logistic Regression: 5 candidates; Random Forest: 12. Preprocessing was fitted inside each pipeline. CV estimates also guided model selection; the held-out test is the final assessment.")
    imp=table("random_forest_importance.csv")
    chart(px.bar(imp.reset_index().rename(columns={imp.index.name or "index":"Feature"}),x="Importance_Percentage",y="Feature",orientation="h",title="Random Forest split-based importance (%)"))
    st.caption("Importance measures model split usage, not causality or percentage of churn explained. It may favour some feature types.")
    with st.expander("Logistic Regression associations and reference categories"):
        st.dataframe(table("logistic_regression_coefficients.csv"),use_container_width=True)
        st.dataframe(pd.read_csv(ROOT/"categorical_reference_groups.csv"),hide_index=True)
        st.write("Numerical coefficients are per one training standard deviation. Categorical odds ratios are relative to the omitted reference group. Odds ratios do not multiply probability; regularised coefficients are not significance tests.")

else:
    st.subheader("Prioritise investigation, then test retention actions")
    recommendations=pd.DataFrame([
        ["Engagement","Viewing hours and login recency: 73.01% of split-based importance","Test relevant content discovery and re-engagement messages","Incremental retention versus a control group"],
        ["Segments","Clusters 1, 4 and 6 have high recorded churn","Investigate recent activity and contact eligibility before targeting","Retention lift and cost per incremental retained customer"],
        ["Payments","Gift Card and Crypto show higher recorded churn","Investigate renewal friction and payment continuity","Payment failure and successful renewal rates"],
        ["Model choice","RF: 96.6% test accuracy, 15 FP and 19 FN","Use as a classroom prioritisation prototype after timing checks","False-positive and false-negative business costs"],
    ],columns=["Focus","Project evidence","Proposed pilot","Measure in a future pilot"])
    st.dataframe(recommendations,hide_index=True,use_container_width=True)
    st.markdown("**Before operational use:** verify dataset provenance, churn-event definition, outcome period, and that predictors were measured before churn. Check calibration and later-period performance. Validate contact eligibility, privacy and intervention costs.")
    st.markdown("**Evidence limits:** no causal retention effect, revenue uplift, ROI or deployed production performance has been demonstrated. The 50.3% churn figure belongs to this sample; these records are not verified Netflix operational data.")
    st.download_button("Download retention pilot proposals",recommendations.to_csv(index=False),"retention_pilot_proposals.csv","text/csv")

st.divider()
st.caption("Group 9 · MBA classroom demonstration · Original fitted models; no model training occurs in this app.")
