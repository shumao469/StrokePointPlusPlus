"""StrokePoint++ reviewed-results viewer. No model loading, training or inference."""
from pathlib import Path
import os
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_core import (ARMS, DISPLAY_NAMES, RARE_BRANCHES, display_subset,
    find_summary, load_public_summary, packaged_images, statistic_text,
    support_text, validate_pointcloud_npz)

RELEASE = Path(__file__).resolve().parents[1]
COLORS = {"BASE3D":"#537E95", "CONCAT":"#A7BCC7", "YZ_UNGATED":"#B9A59A", "YZ_GATE":"#198D85"}


def forest_chart(cohort, key, title):
    figure = go.Figure()
    for arm in reversed(ARMS):
        metric = cohort["original_arm_statistics"][arm][key]
        value = metric.get("value")
        if value is None:
            continue
        interval = metric.get("CI")
        error = dict(type="data", symmetric=False,
            array=[100*(interval[1]-value)], arrayminus=[100*(value-interval[0])],
            thickness=1.6,width=5) if interval else None
        figure.add_trace(go.Scatter(x=[100*value],y=[DISPLAY_NAMES[arm]],mode="markers",
            marker=dict(size=12,color=COLORS[arm]),error_x=error,name=DISPLAY_NAMES[arm],
            hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>"))
    figure.update_layout(title=dict(text=title,font=dict(size=16)),height=285,
        showlegend=False,margin=dict(l=10,r=20,t=45,b=30),
        xaxis_title="%",yaxis_title=None,paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",font=dict(family="Arial",size=12))
    figure.update_xaxes(showgrid=True,gridcolor="#E7EEF1",zeroline=False)
    figure.update_yaxes(automargin=True)
    return figure


def results_tab(cohort):
    arms = cohort["original_arm_statistics"]
    st.caption("Four fixed arms · patient-level uncertainty · no model/threshold selection in this viewer")
    st.dataframe([{"Method":DISPLAY_NAMES[a],
        "Point macro Dice · 95% CI":statistic_text(arms[a]["point_macroDice"]),
        "Native-absence FPR · 95% CI":statistic_text(arms[a]["patientmean_native_absent_FPR"])} for a in ARMS],
        hide_index=True,width="stretch")
    columns = st.columns(2)
    with columns[0]:
        st.plotly_chart(forest_chart(cohort,"point_macroDice","Point-labeling Dice · higher is better"),width="stretch")
    with columns[1]:
        st.plotly_chart(forest_chart(cohort,"patientmean_native_absent_FPR","Native-absence FPR · lower is better"),width="stretch")
    st.markdown("#### Locked paired comparisons")
    rows=[]
    for pair, data in cohort["original_primary_contrasts"].items():
        a,b=pair.split("_minus_")
        rows.append({"Comparison":DISPLAY_NAMES.get(a,a)+" − "+DISPLAY_NAMES.get(b,b),
            "Dice difference":statistic_text(data.get("Dice")).replace("%"," pp",1),
            "FPR difference":statistic_text(data.get("native_absent_FPR")).replace("%"," pp",1)})
    st.dataframe(rows,hide_index=True,width="stretch")
    st.caption("Differences are percentage points. These intervals retain the original comparison-family correction; they are not interchangeable with the marginal intervals above.")
    gate=arms["YZ_GATE"]["correction_and_incremental_error_counts"]["YZ_UNGATED"]
    c=st.columns(4)
    for col,label,value in zip(c,("Corrected points","New error points","Rare TP points lost","Patients modified"),
        (gate["corrected_points"],gate["new_error_points"],gate["rare_true_positive_points_lost"],gate["changed_patients"])):
        col.metric(label,value if not isinstance(value,str) else "N/A")
    if isinstance(gate["changed_patients"],str):
        st.caption("The number of modified patients is not present in the aggregate-only source; it is not inferred from the mean modified-point count.")
    upper=gate.get("population_one_sided95_CP_risk_upper")
    st.warning(f"Incremental-error safety is not certified. The upper 95% patient-event risk bound is {upper:.1%}; 0/20 events still gives 13.9%, not passage of the frozen 10% risk gate.")
    st.caption("Gate counts are relative to the same frozen ungated PointNet++ pipeline—not a clinical error guarantee. No edits or a self-comparison are not evidence of effective safe correction.")


def rare_tab(cohort):
    arm=st.selectbox("Fixed method",ARMS,index=3,format_func=lambda x:DISPLAY_NAMES[x],key="rare_arm")
    rare=cohort["original_arm_statistics"][arm]["rare_branches"]
    records=[]
    for name in RARE_BRANCHES:
        r=rare[name]
        records.append({"Branch":name,"Native + patients":support_text(r["full_native_positive_patients"]),
            "Sampled + patients":r["sampled_positive_patients"],
            "Native + / sample missed":support_text(r["native_positive_but_sampled_zero_patients"]),
            "Native − FPR denominator":support_text(r["absent_FPR_patient_denominator"]),
            "Point recall · CI":statistic_text(r["point_recall"]),
            "Native-absence FPR · CI":statistic_text(r["native_absent_FPR"])})
    st.dataframe(records,hide_index=True,width="stretch")
    st.info("Not sampled ≠ anatomically absent. Undefined class intervals and missing DEV metadata stay unresolved; no labels are read to fill them. Pcom sides remain separate rather than being silently converted to a bilateral OR.")
    st.caption("Recall concerns the 2,048 sampled points. Native presence comes from the full released annotation. These are different denominators; no sampled-point number is presented as full-volume segmentation accuracy.")


def figures_tab():
    main=packaged_images(RELEASE,"figures/main")
    if main:
        for file in main:
            st.image(str(file),caption=file.stem.replace("_"," "),width="stretch")
    else:
        st.info("Reviewed manuscript figure assets will appear here when packaged. No synthetic performance figure is substituted.")
    st.markdown("#### Illustrative cases")
    cases=packaged_images(RELEASE,"assets/cases")
    selected=st.radio("Released anonymous preview",("CaseA","CaseB"),horizontal=True)
    chosen=[file for file in cases if file.stem.lower().startswith(selected.lower())]
    if chosen:
        for file in chosen:
            st.image(str(file),caption=f"{selected} · illustrative research preview; not a new test or diagnostic output",width="stretch")
    else:
        st.caption("Preview not yet packaged. Original patient IDs and private provenance are not displayed.")
    extended=packaged_images(RELEASE,"figures/extended_data")
    if extended:
        with st.expander("Extended-data figures"):
            for file in extended:
                st.image(str(file),caption=file.stem.replace("_"," "),width="stretch")


def local_points_tab():
    st.markdown("#### Optional local point-cloud viewer · no inference")
    st.write("Accepted NPZ: `xyz` or `points` of shape N×3; optional integer `labels` of shape N. At most 8 MiB / 100,000 points; display capped at 25,000 points.")
    if os.getenv("STROKEPOINT_ENABLE_LOCAL_UPLOAD","0") != "1":
        st.info("File selection is disabled by default. An operator may enable it only for a trusted local loopback session; see DEPLOYMENT.md.")
        return
    if os.getenv("TMPDIR") != "/dev/shm":
        st.warning("Local file selection requires the provided launcher with RAM-backed TMPDIR=/dev/shm; otherwise framework multipart parsing could spool temporary files to disk.")
        return
    consent=st.checkbox("I understand: selecting a file transfers it to this local server's session memory, with possible RAM-backed temporary multipart buffering. It is not browser-only processing. The app does not persist a patient dataset, run inference or send the file to third parties.",value=False)
    if not consent:
        st.caption("No upload widget is active. Do not select identifiable or unauthorized patient data.")
        return
    upload=st.file_uploader("Choose a non-identifiable numeric NPZ",type=["npz"],accept_multiple_files=False)
    if upload is None:
        return
    try:
        points,labels=validate_pointcloud_npz(upload.getvalue())
    except (ValueError,MemoryError) as error:
        st.error(str(error));return
    visible,color=display_subset(points,labels)
    marker=dict(size=2.2,opacity=.85,color=color if color is not None else "#198D85")
    if color is not None:
        marker.update(colorscale="Turbo",showscale=True,colorbar=dict(title="Supplied label"))
    figure=go.Figure(go.Scatter3d(x=visible[:,0],y=visible[:,1],z=visible[:,2],mode="markers",marker=marker,hoverinfo="skip"))
    figure.update_layout(height=600,margin=dict(l=0,r=0,t=10,b=0),scene=dict(aspectmode="data"))
    st.plotly_chart(figure,width="stretch")
    st.caption(f"{len(points):,} valid points; {len(visible):,} displayed. Colors, if supplied, are user-provided labels—not model predictions. Bytes may remain in framework/session memory or temporary RAM buffers until cleared. OS-level paging is outside this app's control; secure erasure is not promised.")


def limitations_tab():
    st.markdown("""#### What this research release does—and does not—show

- StrokePoint++-Gate is the frozen Y/Z application: PointNet++ representation, candidate aggregation and an existence gate. It is not a newly validated NC/NMI learning mechanism.
- The task is anatomical identity labeling given a reference-derived binary vessel foreground. It is not automatic vessel extraction, stroke diagnosis, prognosis or treatment advice.
- Lausanne is external DEV; IXI-HH is the locked, record-qualified FINAL cohort. Unknown other-environment use is not “verified unused.” They are shown separately, not pooled into a tuned winner.
- StrokePoint++-3D versus StrokePoint++-Fuse uses matched raw-MRA information and training scale. Comparisons against the frozen Y/Z pipeline differ in inputs and training history.
- The 3D model is a short strong-architecture pilot, not certification of a fully optimized nnU-Net pipeline. The single-MRA Y/Z transfer is not exact replication of its old paired CT/MR experiment.
- Original candidate-identity mapping was unavailable externally. That endpoint remains NOT_COMPUTABLE; sampled-point Dice does not replace or pass its gate.
- Rare branches need their actual patient support and uncertainty. Non-significance is not safety, equivalence or noninferiority.
- Original results, 65-person protected data, Y/Z and STOP are not altered by this viewer. This app performs no model training or inference.
""")
    st.caption("This is a local research-results presentation, not a medical device. Public hosting, datasets and weights need separately reviewed permissions. No account, token or clinical file is required for the reviewed-aggregate view.")


def main():
    st.set_page_config(page_title="StrokePoint++ · Research Explorer",page_icon="🧠",layout="wide")
    st.markdown("<style>.block-container{padding-top:2rem;max-width:1400px}h1{letter-spacing:-.045em} [data-testid=stMetricValue]{font-size:1.7rem}</style>",unsafe_allow_html=True)
    st.title("StrokePoint++")
    st.markdown("**Evidence-guided cerebral vessel labeling · reviewed results explorer**")
    st.caption("研究展示，不用于卒中诊断 · Frozen application evidence, not NC/NMI mechanism certification")
    summary_path=find_summary(RELEASE)
    data=None
    if summary_path:
        try:
            data=load_public_summary(summary_path)
        except (OSError,ValueError,KeyError) as error:
            st.error(f"Reviewed aggregate display is unavailable: {error}")
    else:
        st.info("No reviewed aggregate has been packaged yet. The app will display it after release/results/aggregate/DEV_FINAL_four_arm_summary_v1.json is present.")
    with st.sidebar:
        st.markdown("### Research context")
        st.caption("Gate: frozen Y/Z application\n\n3D/Fuse: fixed comparison models")
        if data:
            cohort_key=st.radio("Cohort",list(data["cohorts"]),format_func=lambda k:k.replace("_"," "))
            cohort=data["cohorts"][cohort_key]
            st.metric("Patients",cohort["n_patients"])
            st.caption("Single MRA · 2,048 foreground points · four fixed arms")
        st.warning("Not for clinical use")
    tabs=st.tabs(("Reviewed results","Rare branches","Figures & cases","Local points","Limitations"))
    with tabs[0]:
        if data:results_tab(cohort)
        else:st.caption("Results require the independently reviewed packaged aggregate; no live artifact or model path is read.")
    with tabs[1]:
        if data:rare_tab(cohort)
        else:st.caption("Rare-branch support will be shown with reviewed data.")
    with tabs[2]:figures_tab()
    with tabs[3]:local_points_tab()
    with tabs[4]:limitations_tab()


if __name__ == "__main__":
    main()
