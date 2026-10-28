"""Reproducible publication artwork from reviewed aggregates and real-case renders.

Scientific graphs, axes, equations and text remain editable in SVG/PDF. Only
actual image/surface/point renderings are raster layers. No training/inference.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import beta

ROOT=Path(__file__).resolve().parents[1]
TEXT='#252A30'; BLUE='#287EAB'; RED='#B83B50'; TEAL='#4D927B'; GOLD='#C58F24'; GRAY='#8B949E'; PURPLE='#9470AD'
ARMS=['BASE3D','CONCAT','YZ_UNGATED','YZ_GATE']
DISPLAY={'BASE3D':'Strong 3D','CONCAT':'Simple fusion','YZ_UNGATED':'Gate off','YZ_GATE':'StrokePoint++-Gate'}
COLORS={'BASE3D':BLUE,'CONCAT':GOLD,'YZ_UNGATED':GRAY,'YZ_GATE':RED}
LABELS=[1,2,3,4,5,6,7,8,9,10,11,12,15]
CLASS=['BA','R-PCA','L-PCA','R-ICA','R-MCA','L-ICA','L-MCA','R-Pcom','L-Pcom','Acom','R-ACA','L-ACA','3rd-A2']
CAPTIONS={}
SOURCES=[]
ALL=[]
CROP=None
CLASS_COLORS=['#7b3294','#c51b7d','#de77ae','#2166ac','#4393c3','#b2182b','#d6604d','#1b9e77','#66a61e','#e6ab02','#7570b3','#a6761d','#e7298a']


def configure():
    global CROP
    # Optional authorized system-font directory, not a bundled font or private path.
    font_dir=os.environ.get('STROKEPOINT_FONT_DIR')
    if font_dir:
        for name in ('arial.ttf','arialbd.ttf'):
            p=Path(font_dir)/name
            if p.exists(): font_manager.fontManager.addfont(str(p))
    family='Arial' if any(v.name=='Arial' for v in font_manager.fontManager.ttflist) else 'DejaVu Sans'
    plt.rcParams.update({'font.family':family,'font.size':8,'axes.titlesize':9,'axes.labelsize':8,
        'xtick.labelsize':7.2,'ytick.labelsize':7.2,'axes.edgecolor':TEXT,'text.color':TEXT,
        'axes.labelcolor':TEXT,'xtick.color':TEXT,'ytick.color':TEXT,'axes.linewidth':.65,
        'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white',
        'figure.facecolor':'white','lines.linewidth':1.0,'mathtext.fontset':'dejavusans'})
    bounds=[]
    for p in (ROOT/'assets/cases').glob('Case*_*.png'):
        if p.stem.endswith(('mra','surface')): continue
        z=plt.imread(p)[...,:3]
        ys,xs=np.where(np.any(z<.94,axis=2))
        if len(xs): bounds.append((int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())))
    if bounds:
        CROP=(max(0,min(v[0] for v in bounds)-28),max(0,min(v[1] for v in bounds)-28),max(v[2] for v in bounds)+29,max(v[3] for v in bounds)+29)


def new(h=235): return plt.figure(figsize=(180/25.4,h/25.4))


def panel(fig,rect,letter,title,off=False):
    ax=fig.add_axes(rect)
    if letter:
        fig.text(rect[0]-.012,rect[1]+rect[3]+.016,letter,fontsize=11,fontweight='bold',va='bottom')
    if title: fig.text(rect[0]+.024,rect[1]+rect[3]+.018,title,fontsize=9,va='bottom')
    for s in ('top','right'): ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5,width=.65)
    if off: ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    return ax


def note(fig,text,y=.017): fig.text(.045,y,text,fontsize=6.8,color='#52647F',va='bottom')


def vessel_legend(fig,y):
    handles=[Line2D([],[],ls='',marker='o',ms=3.5,color=c) for c in CLASS_COLORS]
    fig.legend(handles,CLASS,loc='upper left',bbox_to_anchor=(.042,y),ncol=7,frameon=False,
               fontsize=6.3,handlelength=.7,handletextpad=.3,columnspacing=1.05,borderaxespad=0)


def photo(ax,name,title=None,crop=None):
    p=ROOT/'assets/cases'/name
    if not p.exists(): raise FileNotFoundError('Required real-case render missing: '+str(p))
    pixels=plt.imread(p)
    if crop=='axial':
        pixels=pixels[:,2*pixels.shape[1]//3:,:]
    elif not name.endswith(('mra.png','surface.png')) and CROP:
        x0,y0,x1,y1=CROP;pixels=pixels[y0:y1,x0:x1]
    ax.imshow(pixels)
    ax.set_xlim(-.5,pixels.shape[1]-.5)
    ax.set_ylim(pixels.shape[0]-.5,-.5)
    ax.set_aspect('equal');ax.axis('off')
    if title: ax.set_title(title,fontsize=8,pad=2)


def box(ax,xy,w,h,text,color=BLUE,fontsize=8,fill=True):
    ax.add_patch(Rectangle(xy,w,h,facecolor=(color+'10') if False else ('#EEF5F9' if color==BLUE else '#FBF1F3' if color==RED else '#F3F7F5' if color==TEAL else '#FAF5E9'),edgecolor=color,lw=.75))
    ax.text(xy[0]+w/2,xy[1]+h/2,text,ha='center',va='center',fontsize=fontsize,color=TEXT,linespacing=1.3)


def arrow(ax,a,b,color=GRAY,style='-'):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,color=color,lw=.8,linestyle=style,shrinkA=1.5,shrinkB=1.5))


def errpoint(ax,y,val,ci,color=BLUE,marker='o',size=4):
    if ci is not None and all(v is not None and np.isfinite(v) for v in ci):
        ax.plot(ci,[y,y],color=color,lw=1.2,zorder=2)
        ax.plot([ci[0],ci[0]],[y-.045,y+.045],color=color,lw=.7)
        ax.plot([ci[1],ci[1]],[y-.045,y+.045],color=color,lw=.7)
    if val is not None and np.isfinite(val): ax.plot(val,y,marker=marker,color=color,ms=size,zorder=3)


def legacy_plot(ax,statistics,metric,arms,names,colors):
    for i,arm in enumerate(arms):
        z=statistics[arm][metric];errpoint(ax,len(arms)-1-i,z['mean'],[z['ci95_lo'],z['ci95_hi']],colors[i])
        SOURCES.append(dict(figure='internal',group=arm,metric=metric,value=z['mean'],lo=z['ci95_lo'],hi=z['ci95_hi'],n=z['n']))
    ax.set_yticks(range(len(arms)),names[::-1]);ax.set_ylim(-.5,len(arms)-.5)
    ax.set_xlabel('Native-absence FPR' if metric=='absent_FPR' else 'Point macro-Dice')
    ax.grid(axis='x',color='#E3E6E9',lw=.5,zorder=0)


def ext_plot(ax,s,metric):
    for i,a in enumerate(ARMS):
        z=s['summary'][a][metric];errpoint(ax,3-i,z['value'],z['CI'],COLORS[a])
        SOURCES.append(dict(figure='external',group=s['external_role']+'_'+a,metric=metric,value=z['value'],lo=z['CI'][0] if z['CI'] else '',hi=z['CI'][1] if z['CI'] else '',n=s['n_patients'] if metric=='Dice' else s['FPR_valid_patients']))
    ax.set_yticks(range(4),[DISPLAY[a] for a in ARMS[::-1]]);ax.set_ylim(-.5,3.5)
    ax.set_xlabel('Point macro-Dice' if metric=='Dice' else 'Native-absence FPR')
    ax.grid(axis='x',color='#E3E6E9',lw=.5,zorder=0)
    ax.set_xlim((.53,.94) if metric=='Dice' else (-.015,.58))


def save(fig,folder,name,caption):
    out=ROOT/'figures'/folder;out.mkdir(parents=True,exist_ok=True)
    stem=out/name
    fig.savefig(stem.with_suffix('.svg'),metadata={'Creator':'StrokePoint++ reviewed-evidence figure builder'})
    fig.savefig(stem.with_suffix('.pdf'),metadata={'Title':name,'Author':'StrokePoint++ research package'})
    fig.savefig(stem.with_suffix('.png'),dpi=450)
    ALL.append(stem.with_suffix('.pdf'));CAPTIONS[name]=caption
    plt.close(fig)


def fig1(D):
    f=new(235)
    photo(panel(f,[.04,.705,.28,.245],'a','Source MRA',True),'CaseA_mra.png',crop='axial')
    photo(panel(f,[.36,.705,.28,.245],'b','Given vessel foreground',True),'CaseA_surface.png')
    photo(panel(f,[.68,.705,.28,.245],'c','Reference identity colors',True),'CaseA_truth.png')
    vessel_legend(f,.718)
    ax=panel(f,[.04,.28,.92,.36],'d','StrokePoint++-Gate: implemented workflow',True)
    box(ax,(.005,.56),.16,.26,'Binary vessel\nforeground\n+ 2,048 points',BLUE,8)
    box(ax,(.225,.56),.20,.26,'PointNet++-derived\nFullInformationPP\nFrozen encoder',BLUE,8)
    box(ax,(.49,.56),.195,.26,'Calibrated logits\n+ point features\nF, L: 64 channels',BLUE,8)
    box(ax,(.755,.56),.235,.26,'13-class point\nidentity prediction',BLUE,8)
    for a,b in [((.165,.69),(.225,.69)),((.425,.69),(.49,.69)),((.685,.69),(.755,.69))]:arrow(ax,a,b,BLUE)
    box(ax,(.22,.075),.225,.255,'Predicted segments\nDBSCAN + mean/max\n256-D descriptor',TEAL,8)
    box(ax,(.505,.075),.215,.255,'Two fitted LR heads\nPcom / third-A2\nSaved scaler + weights',RED,8)
    box(ax,(.78,.075),.21,.255,'Presence gate\nReassign blocked\nclass predictions',RED,8)
    arrow(ax,(.58,.56),(.58,.43),TEAL);arrow(ax,(.58,.43),(.332,.43),TEAL);arrow(ax,(.332,.43),(.332,.33),TEAL)
    arrow(ax,(.445,.202),(.505,.202),TEAL);arrow(ax,(.72,.202),(.78,.202),RED)
    arrow(ax,(.88,.56),(.88,.33),RED)
    ax.text(.01,.01,'Given-foreground anatomical labeling. No image-to-vessel extraction or stroke-risk prediction.',fontsize=7.2,color=GRAY)
    ax=panel(f,[.04,.065,.92,.15],'e','Separate evidence roles and versions',True)
    for x,title,detail,col in [(0,'Original application','48 fit / 65 internal patients\nPaired CT/MR aggregation',BLUE),(.34,'Single-MRA reconstruction','Fixed R0v2 recipe\nNo recovery of original LR states',TEAL),(.68,'External fixed evaluation','Lausanne DEV: 20\nIXI-HH FINAL: 20',RED)]:
        ax.text(x,.82,title,fontsize=8.5,fontweight='bold',color=col)
        ax.text(x,.49,detail,fontsize=7.7,va='top',linespacing=1.4)
    note(f,'Deployment geometry is label-blind; reference identity colors and source MRA are shown for display, not as Gate inputs.')
    save(f,'main','fig1_framework','Figure 1 | StrokePoint++ study and implemented application. a-c, Real MRA, supplied binary vessel surface and corresponding labeled point cloud from the first frozen Lausanne DEV case. d, Source-verified frozen PointNet++-derived representation, calibrated identity readout, predicted-segment aggregation, two logistic presence heads and blocked-class reassignment. The heads do not certify pointwise or patient-event safety. e, Original paired CT/MR application, separately reconstructed single-MRA interface and fixed external DEV/FINAL roles. The task conditions on a supplied vessel foreground; neither end-to-end vessel extraction nor clinical stroke prediction has been established. The spatial visual narrative is inspired by Dai et al. (2026), not an implementation of their OT mechanism.')


def fig2(D):
    f=new(240)
    for x,name,title,letter in [(.04,'CaseA_input.png','Input: 2,048 points','a'),(.36,'CaseA_grouping.png','SA1: 128 anchors + grouping','b'),(.68,'CaseA_s2.png','SA2: 32 anchors','c')]:
        photo(panel(f,[x,.75,.28,.20],letter,title,True),name)
    for x,col,label in [(.06,'#b7bcc3','Source points'),(.30,BLUE,'SA1 anchors'),(.52,GOLD,'Selected neighborhood'),(.81,RED,'Query / SA2')]:
        f.text(x,.738,'\u25cf',color=col,fontsize=8.2)
        f.text(x+.018,.738,label,fontsize=7.0,color=TEXT)
    ax=panel(f,[.04,.39,.92,.30],'d','Frozen hierarchical representation and saved readout',True)
    ys=.64; xs=[.0,.21,.42,.63,.84];ww=.16
    texts=['Relative xyz\nShared MLP\n3 -> 32 -> 64','SA2 + local route\n+ shared native\npath descriptors','Global context\n131 -> 128 -> 256\nmax aggregation','Feature propagation\nFP2: 448 -> 128\nFP1: 131 -> 64','Saved readout\n+ fixed priors\n13 identity logits']
    for x,t in zip(xs,texts):box(ax,(x,ys),ww,.28,t,BLUE,7.2)
    for x in xs[:-1]:arrow(ax,(x+ww,ys+.14),(x+.21,ys+.14),BLUE)
    arrow(ax,(.08,.64),(.08,.54),BLUE)
    arrow(ax,(.08,.54),(.71,.54),BLUE)
    arrow(ax,(.71,.54),(.71,.64),BLUE)
    ax.text(.30,.495,'SA1 skip features to FP2',fontsize=6.8,color=BLUE)
    ax.text(.02,.40,'FullInformationPP shares native path-distance information across its information-matched controls.',fontsize=7.6,color=GRAY)
    box(ax,(.02,.04),.34,.23,'F and L point features\n64 + 64 dimensions',TEAL,8)
    box(ax,(.44,.04),.54,.23,'Candidate descriptor\nh = [mean(F), max(F), mean(L), max(L)]\n256 dimensions',TEAL,8)
    arrow(ax,(.36,.155),(.44,.155),TEAL)
    ax=panel(f,[.04,.10,.92,.225],'e','Candidate-to-patient presence decision and execution',True)
    box(ax,(.0,.51),.28,.31,'Predicted candidate segments\nDBSCAN: eps = 0.015\nmin_samples = 3',TEAL,8)
    box(ax,(.36,.51),.29,.31,'Max: candidates / sides\nPaired CT/MR: max\nExternal transfer: MRA only',TEAL,7.5)
    box(ax,(.73,.51),.27,.31,'p = sigmoid(w.T z + b)\nz: saved fit standardization\nFixed application tau = 0.05',RED,7.8)
    arrow(ax,(.28,.665),(.36,.665),TEAL);arrow(ax,(.65,.665),(.73,.665),RED)
    ax.text(.0,.28,'If presence evidence falls below the saved threshold:',fontsize=8)
    ax.text(.02,.10,'block Pcom labels 8/9 or third-A2 label 15; reassign only affected argmax predictions.',fontsize=8,color=RED)
    note(f,'Candidate generation and geometric neighborhoods are different operations. Spatial proximity is not vessel connectivity.')
    save(f,'main','fig2_architecture_algorithm','Figure 2 | Source-verified PointNet++ representation and candidate decision. a-c, Label-blind input geometry and fixed anchor subsets from the same real DEV foreground, not learned attention or connectivity labels. The displayed SA1 query has 32 neighborhood slots but only 21 unique source points, including repeated self-padding; it is not 32 distinct neighbors. d, Hierarchical local/context paths, shared full native path descriptors, feature propagation and saved identity readout. F/L descriptor pooling follows the frozen implementation. e, DBSCAN operates on normalized xyz of predicted-class points. Segment descriptors are max-aggregated, standardized by saved training statistics and passed to logistic heads. The application threshold is fixed at 0.05. The original internal version pools paired CT/MR; the reconstructed external version deploys MRA alone and is not the original model state. Reassignment is not an explicit unknown state, and a presence score is not an action-safety probability. No OT, active reading or new joint objective is claimed.')


def fig3(D):
    f=new(220)
    arms=['perm_cal','real_cal','clf_structural','clf_embedding','oracle_fair']
    names=['No-local control','Local + calibration','Structural + LR','Embedding + LR','GT-assisted oracle']
    cols=[GRAY,BLUE,GOLD,RED,PURPLE]
    a=panel(f,[.28,.625,.28,.29],'a','Development: FPR')
    legacy_plot(a,D['internal_development']['arm_statistics'],'absent_FPR',arms,names,cols);a.set_xlim(-.015,.35)
    b=panel(f,[.68,.625,.28,.29],'b','Development: Dice')
    legacy_plot(b,D['internal_development']['arm_statistics'],'Dice',arms,names,cols);b.set_yticklabels([]);b.set_xlim(.59,.70)
    a=panel(f,[.28,.23,.28,.25],'c','Original internal: FPR')
    yy=D['internal_validation']['arm_statistics'];use=['real_cal','clf_patient_level','oracle_fair']
    legacy_plot(a,yy,'absent_FPR',use,['Local + calibration','StrokePoint++-Gate','GT-assisted oracle'],[BLUE,RED,PURPLE]);a.set_xlim(-.005,.19)
    b=panel(f,[.68,.23,.28,.25],'d','Original internal: Dice')
    legacy_plot(b,yy,'Dice',use,['Local + calibration','StrokePoint++-Gate','GT-assisted oracle'],[BLUE,RED,PURPLE]);b.set_yticklabels([]);b.set_xlim(.62,.705)
    f.text(.28,.545,'Exploratory development, 60 patients; FPR valid n = 58',fontsize=7.6,color=GRAY)
    f.text(.28,.155,'Internal held-out patients: Dice n = 65; FPR valid n = 64',fontsize=7.6,color=GRAY)
    f.text(.28,.11,'FPR 0.1257 -> 0.0710: 43.5% relative reduction',fontsize=9,color=RED)
    note(f,'Original paired CT/MR application only. Historical audit limitations retained. Oracle uses target information and is not deployable.')
    save(f,'main','fig3_internal_evidence','Figure 3 | Original development and internal application evidence. a,b, FP0-X historical clean-benchmark aggregates, with original 95% intervals and effective denominators; development comparisons are exploratory. A recorded historical reviewer-source hash mismatch remains an audit limitation and is not repaired by relabeling these outputs. c,d, Original FP0-Y held-out paired CT/MR application: 65 patients contribute Dice, while 64 with at least one absent native class contribute FPR. FPR changes from 0.125651 to 0.070964, a 43.52% relative reduction. Oracle arms are target-assisted diagnostics, not deployed comparators or information-theoretic limits. These are not the reconstructed single-MRA external model states. No raw data from the protected 65 were accessed for artwork; only existing aggregate JSON was used.')


def fig4(D):
    f=new(245)
    for role,y,letter in [('DEV',.735,'a'),('FINAL',.48,'c')]:
        s=D['external'][role];cohort='Lausanne DEV' if role=='DEV' else 'IXI-HH FINAL'
        ax=panel(f,[.265,y,.30,.18],letter,cohort+' | Dice');ext_plot(ax,s,'Dice')
        ax=panel(f,[.68,y,.28,.18],chr(ord(letter)+1),cohort+' | FPR');ext_plot(ax,s,'native_absent_FPR');ax.set_yticklabels([])
    ax=panel(f,[.04,.075,.92,.31],'e','Same-case anatomical outputs: fixed DEV Case B',True)
    for i,(name,title) in enumerate([('truth','Reference'),('BASE3D','Strong 3D'),('CONCAT','Simple fusion'),('YZ_GATE','StrokePoint++-Gate')]):
        sub=ax.inset_axes([i*.25,.09,.245,.83]);photo(sub,'CaseB_'+name+'.png',title)
    vessel_legend(f,.075)
    note(f,'Cohorts remain separate. Fixed model checkpoints and thresholds; no FINAL tuning. Point macro-Dice, not full-volume Dice.')
    save(f,'main','fig4_external_validation','Figure 4 | Locked external single-MRA evaluation. a-d, Four fixed arms on 20 Lausanne DEV and 20 IXI-HH FINAL patients, kept separate rather than pooled. Points and intervals are descriptive patient-level means and arm-wise 95% bootstrap intervals; primary paired family-adjusted contrasts are reported separately in Extended Data Figure 3. FPR is a patient mean over evaluable native class-absence indicators: 20 evaluable DEV patients and 19 FINAL patients. e, The second frozen DEV case, selected by manifest order independently of performance, shown with an identical view and point correspondence. BASE3D and CONCAT share raw-image inputs and short-pilot training scale; Y/Z differs in information and training history, so cross-family rankings do not isolate pure architectural causation. Strong3D is not a fully optimized nnU-Net pipeline. FINAL is source-record-qualified within this study, not proof of globally never-used data.')


def recall_heat(ax,s,metric='recall',rare_only=False):
    labs=[8,9,10,15] if rare_only else LABELS
    arr=np.array([[next(c for c in s['per_class'][a] if c['label']==lab)[metric]['value'] for lab in labs] for a in ARMS],float)
    im=ax.imshow(arr,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    for i in range(4):
        for j in range(len(labs)):
            z=arr[i,j];ax.text(j,i,'NA' if not np.isfinite(z) else f'{z:.2f}',ha='center',va='center',color='white' if z>.64 else TEXT,fontsize=7.5 if len(labs)==4 else 6.5)
    ax.set_yticks(range(4),[DISPLAY[a] for a in ARMS]);ax.set_xticks(range(len(labs)),[CLASS[LABELS.index(l)] for l in labs],rotation=0 if rare_only else 50,ha='center' if rare_only else 'right')
    ax.tick_params(length=0,pad=5)
    for sp in ax.spines.values():sp.set_visible(False)
    return im


def fig5(D):
    f=new(235);s=D['external']['FINAL']
    a=panel(f,[.27,.72,.69,.20],'a','FINAL rare-vessel point recall');recall_heat(a,s,rare_only=True)
    a.set_xticklabels(['R-Pcom\n(n = 11)','L-Pcom\n(n = 11)','Acom\n(n = 19)','3rd-A2\n(n = 2)'])
    a.text(0,-.36,'Native-positive patient support',transform=a.transAxes,fontsize=7.2,color=GRAY)
    a=panel(f,[.12,.385,.34,.22],'b','Gate edits versus its own output')
    totals=s['gate_correction_totals'];v=[totals['corrected'],totals['new_errors'],totals['rare_TP_lost']]
    bars=a.bar(range(3),v,color=[TEAL,RED,GOLD],width=.57)
    for b,x in zip(bars,v):a.text(b.get_x()+b.get_width()/2,x+.2,str(x),ha='center',fontsize=9)
    a.set_xticks(range(3),['Corrected','New errors','Rare TP lost'],rotation=18,ha='right');a.set_ylabel('Number of points');a.set_ylim(0,10)
    a.text(0,-.35,'31 modified points across 5 patients\nRare TP loss overlaps the new-error count.',transform=a.transAxes,fontsize=7.3,color=GRAY,va='top')
    a=panel(f,[.63,.385,.33,.22],'c','Patient-event risk precision')
    nn=np.arange(4,81);zero=1-.05**(1/nn);one=beta.ppf(.95,2,nn-1)
    a.plot(nn,zero,color=BLUE,label='0 events');a.plot(nn,one,color=RED,label='1 event')
    a.axhline(.1,color=TEXT,lw=.8,ls='--',label='Original 10% cap');a.scatter([20],[beta.ppf(.95,2,19)],color=RED,s=23,zorder=4)
    a.annotate('FINAL: 1 / 20\nupper = 21.6%',(20,beta.ppf(.95,2,19)),xytext=(35,.35),fontsize=7.6,arrowprops={'arrowstyle':'-','lw':.65,'color':RED})
    a.set(xlim=(4,80),ylim=(0,.65),xlabel='Number of patients',ylabel='One-sided 95% risk upper bound');a.legend(frameon=False,fontsize=7,loc='upper right')
    a=panel(f,[.08,.07,.88,.15],'d','What the present evidence establishes',True)
    a.text(.0,.85,'Application signal',color=TEAL,fontweight='bold',fontsize=9)
    a.text(.0,.59,'Gate reduces observed false identity presence,\nwith a residual rare-point error.',fontsize=8,va='top',linespacing=1.4)
    a.text(.55,.85,'Unresolved target',color=RED,fontweight='bold',fontsize=9)
    a.text(.55,.59,'Acom identity and third-A2 recovery remain weak.\nNeither safety nor new-mechanism superiority passed.',fontsize=8,va='top',linespacing=1.4)
    note(f,'No equivalence or safety claim from a zero-crossing interval. Acom negatives n = 1; third-A2 positives n = 2.')
    save(f,'main','fig5_rare_branches_safety','Figure 5 | Rare-branch performance and explicit correction safety. a, Reviewed FINAL point recall for each method; native-positive supports are 11, 11, 19 and 2 for right Pcom, left Pcom, Acom and third-A2. Small-support undefined intervals remain undetermined. b, Gate relative to its own ungated reconstruction: 8 corrected points, 1 new error and 1 overlapping rare true-positive loss among 31 modified points in 5 patients. These must not be conflated with cross-model discrepancies against BASE3D. c, Formula-based Clopper-Pearson precision curves, not additional patients or experiments. The actual 1/20 event result has a one-sided 95% upper bound of 0.2161, exceeding the unchanged 0.10 cap. d, Application utility and remaining scientific limits. Presence gating is not a reliable action-risk certificate.')


def ed1(D):
    f=new(220)
    photo(panel(f,[.05,.64,.41,.29],'a','Native MRA crop',True),'CaseA_mra.png')
    photo(panel(f,[.55,.64,.41,.29],'b','Input-only vessel surface',True),'CaseA_surface.png')
    ax=panel(f,[.05,.30,.91,.26],'c','Coordinate and input contract',True)
    for x,t in [(0,'Native voxel index\nNIfTI affine, millimetres'),(.35,'Common world point\nMatched image orientation'),(.70,'Shared sampled index\nAll arms, same correspondence')]:box(ax,(x,.4),.29,.34,t,BLUE,7.7)
    arrow(ax,(.29,.57),(.35,.57),BLUE);arrow(ax,(.64,.57),(.70,.57),BLUE)
    ax.text(.0,.15,'Only the binary union conditions preprocessing. Class identities remain evaluation targets.',fontsize=8)
    ax=panel(f,[.05,.07,.91,.15],'d','Sampling does not establish anatomical absence',True)
    box(ax,(.0,.18),.45,.55,'Full native presence\nUsed for absent-class denominators',TEAL,8)
    box(ax,(.55,.18),.45,.55,'Sampled point labels\nUsed for point recall and Dice',GOLD,8)
    note(f,'All examples are fixed DEV illustrations, not augmentation experiments or a clinical segmentation workflow.')
    save(f,'extended_data','edfig1_input_correspondence','Extended Data Figure 1 | Input geometry and conditional task. a,b, The real DEV input and binary vessel surface. c, Native affine, physical coordinate and fixed point-index correspondence are kept explicit; all methods share the same sampled points. d, Full-mask anatomical presence and sampled-point support are distinct. A missing sampled label does not establish anatomical absence. Panels are diagrams of checked operations, not new model experiments.')


def ed2(D):
    f=new(190)
    for role,y,letter in [('DEV',.60,'a'),('FINAL',.18,'b')]:
        s=D['external'][role];ax=panel(f,[.235,y,.72,.25],letter,role+' | full 13-class point recall');recall_heat(ax,s)
    note(f,'Canonical labels include third-A2 = 15. Empty-support intervals remain undefined rather than silently zero-filled.')
    save(f,'extended_data','edfig2_full_class_recalls','Extended Data Figure 2 | Complete 13-class external point-recall descriptors. a,b, Lausanne DEV and IXI-HH FINAL remain separate. Entries are reviewed per-patient positive-support recall means, not pooled point accuracy. Class ordering follows the native canonical set 1-12,15. See source data for supports, full-native positives/negatives and interval status. No missing interval is converted into evidence of equivalence.')


def ed3(D):
    f=new(195)
    pairs=['CONCAT_minus_BASE3D','YZ_GATE_minus_YZ_UNGATED','YZ_GATE_minus_BASE3D']
    names=['Fusion - Strong 3D','Gate - Gate off','Gate - Strong 3D']
    for role,y,letter in [('DEV',.60,'a'),('FINAL',.19,'c')]:
        s=D['external'][role]
        for k,(metric,x) in enumerate([('Dice',.33),('native_absent_FPR',.72)]):
            ax=panel(f,[x,y,.24,.25],chr(ord(letter)+k),role+' | '+('Dice delta' if k==0 else 'FPR delta'))
            for i,pair in enumerate(pairs):
                z=s['contrasts'][pair][metric];errpoint(ax,2-i,z['value'],z['CI'],[GOLD,RED,BLUE][i])
            ax.set_yticks(range(3),names[::-1] if k==0 else ['']*3);ax.axvline(0,color=GRAY,lw=.7,ls='--');ax.set_ylim(-.4,2.4)
            ax.set_xlim((-.25,.08) if k==0 else (-.5,.5));ax.set_xlabel('Paired patient-level difference')
    note(f,'Original simultaneous family alpha = 0.05 / 3. Intervals include pairing; sign alone is not statistical superiority.')
    save(f,'extended_data','edfig3_paired_contrasts','Extended Data Figure 3 | Prespecified paired external comparisons. a-d, Cohort-specific differences use the original patient-paired bootstrap and simultaneous three-comparison intervals. Negative Dice differences indicate lower point macro-Dice, while negative FPR differences indicate fewer absent-class false detections. FINAL fusion Dice and gate-versus-own improvements have intervals crossing or touching zero. These panels do not change comparators or thresholds after inspecting FINAL.')


def ed4(D):
    f=new(195);s=D['external']['FINAL']
    ax=panel(f,[.16,.60,.80,.27],'a','FINAL native and sampled class support')
    pos=[c['full_native_positive_patients'] for c in s['per_class']['BASE3D']];neg=[c['full_native_negative_patients'] for c in s['per_class']['BASE3D']]
    ax.bar(np.arange(13),pos,color=BLUE,label='Native positive');ax.bar(np.arange(13),neg,bottom=pos,color='#DFE8EE',label='Native absent')
    ax.set_xticks(range(13),CLASS,rotation=50,ha='right');ax.set(ylim=(0,22),ylabel='Patients');ax.legend(frameon=False,ncol=2,fontsize=7)
    ax=panel(f,[.08,.16,.40,.25],'b','Sparse-support caveats',True)
    ax.text(.0,.9,'Acom',fontsize=10,fontweight='bold',color=RED);ax.text(.0,.68,'19 native-positive patients\n1 native-negative patient\nAbsence-FPR uncertainty unresolved',fontsize=8,va='top',linespacing=1.5)
    ax=panel(f,[.57,.16,.39,.25],'c','Metric denominators',True)
    ax.text(.0,.9,'Third-A2',fontsize=10,fontweight='bold',color=GOLD);ax.text(.0,.68,'2 native-positive patients\n63 sampled reference points\nPoint-recall interval undetermined',fontsize=8,va='top',linespacing=1.5)
    note(f,'Overall Dice: n = 20. Overall absent FPR: n = 19. Patient-event risk: n = 20. These are different denominators.')
    save(f,'extended_data','edfig4_support_precision','Extended Data Figure 4 | Support and denominator limits. a, Complete native positive and absent patient counts for all canonical classes in FINAL. b,c, Sparse Acom-negative and third-A2-positive supports preclude precise rare-branch guarantees. One FINAL patient has no absent native class, making overall FPR undefined for that patient; the patient remains in Dice and risk-event analyses. Undefined bootstrap draws are preserved as an undetermined interval, not discarded to create a precise unconditional estimate.')


def supp1(D):
    f=new(250)
    for r,case in enumerate(['CaseA','CaseB','CaseC','CaseD']):
        for c,(arm,title) in enumerate([('truth','Reference'),('BASE3D','Strong 3D'),('CONCAT','Fusion'),('YZ_GATE','Gate')]):
            ax=panel(f,[.045+c*.24,.745-r*.225,.225,.18],chr(ord('a')+r) if c==0 else '',case+' | '+title if c==0 else title,True)
            photo(ax,case+'_'+arm+'.png')
    vessel_legend(f,.065)
    note(f,'First four frozen Lausanne DEV cases. Identical within-case camera and point mapping; not selected for model quality.')
    save(f,'supplementary','suppfig1_real_pointcloud_gallery','Supplementary Figure 1 | Deterministic real-case point-cloud gallery. The first four frozen Lausanne DEV cases appear in manifest order, independent of outcome. Within each case, reference and all model outputs share points, projection and scale. Colors denote anatomical identity, not model attention. No FINAL case was selected or tuned for this gallery.')


def supp2(D):
    f=new(200);s=D['external']['FINAL']
    for j,metric in enumerate(['recall','Dice','precision']):
        ax=panel(f,[.25,.72-j*.28,.71,.17],chr(ord('a')+j),'FINAL rare classes | '+metric)
        recall_heat(ax,s,metric,rare_only=True)
    note(f,'Means are descriptive with their own eligibility sets; see source JSON for CI status and exact denominators.')
    save(f,'supplementary','suppfig2_rare_metric_profiles','Supplementary Figure 2 | Rare-class metric profiles in FINAL. Recall, class Dice and precision use their corresponding eligibility conditions and must not be treated as interchangeable measures. Means with sparse support remain descriptive; the exact confidence-interval status and effective denominators are provided in source data. No aggregate FPR reduction substitutes for rare true-positive preservation.')


def supp3(D):
    f=new(200);ax=panel(f,[.07,.12,.89,.78],'a','Version and evidence separation',True)
    items=[('Gate-AppYZ-v1','Original paired CT/MR application','Internal aggregate evidence; original Y/Z preserved',BLUE),('Gate-MRA-R0v2','Fixed-recipe reconstruction','Single-MRA external transfer, not original LR states',RED),('StrokePoint-3D','Raw MRA + strong 3D baseline','Short fixed pilot; full nnU-Net optimization not claimed',TEAL),('StrokePoint-Fuse','Simple image / point fusion control','Same input comparison, no superiority established',GOLD),('Research / archive','OT, routing, value and risk-head attempts','Negative / invalid / stopped versions remain separate',GRAY)]
    for i,(name,sub,detail,col) in enumerate(items):
        y=.94-i*.19;ax.text(0,y,name,color=col,fontsize=10,fontweight='bold');ax.text(0,y-.055,sub,fontsize=8.4);ax.text(0,y-.105,detail,fontsize=7.6,color=GRAY)
        if i<4:ax.plot([0,1],[y-.145,y-.145],color='#E3E6E9',lw=.65)
    note(f,'Archive integrity is not scientific admission. Protected data, original thresholds and STOP remain unchanged.')
    save(f,'supplementary','suppfig3_versions_and_limits','Supplementary Figure 3 | Version and evidence taxonomy. Original application, newly reconstructed single-MRA transfer, strong baseline, simple fusion and research variants are named separately. Contaminated or implementation-invalid historical results do not support a positive claim, and stopped mechanisms do not become successful through renaming. The code package records executable boundaries, model hashes and third-party terms; it does not redistribute raw patient images, labels or weights.')


def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args()
    ROOT=args.root.resolve()
    configure();D=json.loads((ROOT/'source_data/figure_data.json').read_text())
    for fn in [fig1,fig2,fig3,fig4,fig5,ed1,ed2,ed3,ed4,supp1,supp2,supp3]:fn(D)
    hero=plt.figure(figsize=(12,4.15))
    hero.text(.045,.73,'StrokePoint++',fontsize=33,fontweight='bold',color=TEXT)
    hero.text(.045,.56,'Cerebral vessel identity with PointNet++',fontsize=14,color=BLUE)
    hero.text(.045,.37,'Frozen point representations and candidate evidence\nwith explicit baseline, rare-branch and safety evaluation',fontsize=11.2,color=TEXT,linespacing=1.6)
    hero.text(.045,.12,'Research software  |  Given vessel foreground  |  Not a stroke diagnostic device',fontsize=8,color=GRAY)
    image_ax=hero.add_axes([.61,.02,.37,.96]);photo(image_ax,'CaseA_surface.png')
    (ROOT/'assets').mkdir(exist_ok=True)
    hero.savefig(ROOT/'assets/README_hero.png',dpi=160)
    hero.savefig(ROOT/'assets/README_hero.svg');plt.close(hero)
    import fitz
    merged=fitz.open()
    for path in ALL:
        with fitz.open(path) as pdf: merged.insert_pdf(pdf)
    combined=ROOT/'figures/StrokePointPlusPlus_FigureBook.pdf';merged.save(combined);merged.close()
    (ROOT/'figures/FIGURE_LEGENDS.md').write_text('# StrokePoint++ figure legends\n\n'+'\n\n'.join(CAPTIONS.values())+'\n\nReferences: Dai et al., Nature Methods 23:760-771 (2026), doi:10.1038/s41592-026-03034-9. Yang et al., NEJM AI 3(8):AIdbp2500994 (2026), doi:10.1056/AIdbp2500994. TopCoW release, doi:10.5281/zenodo.15692630. Raw Lausanne images CC0; manual annotations follow separately recorded TopCoW attribution/noncommercial terms. Reference illustrations are original redraws, not copied third-party figures.\n',encoding='utf-8')
    with (ROOT/'source_data/figure_estimates.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['figure','group','metric','value','lo','hi','n']);writer.writeheader();writer.writerows(SOURCES)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'figures').rglob('*') if p.is_file() and p.name!='artwork_manifest.json'}
    (ROOT/'figures/artwork_manifest.json').write_text(json.dumps({'main_figures':5,'extended_data_figures':4,'supplementary_figures':3,'outputs':hashes,'raster_layers':'Only actual data images/surface/point renders','text_and_scientific_graphics_editable_in_SVG':True,'new_fits':0,'new_forwards':0},indent=2),encoding='utf-8')
    print(json.dumps({'main':5,'extended':4,'supplementary':3,'combined':str(combined)}))


if __name__=='__main__':main()
