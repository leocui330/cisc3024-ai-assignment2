"""Recompute final metrics and figures from saved predictions; no training."""
import csv
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / "temp" / "matplotlib")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from metrics import metrics

RUN=ROOT/"output"/"runs"
PUBLIC=ROOT/"results"
FIG=ROOT/"assets"
NAMES={"map":"MAP", "map_temperature":"MAP + temperature", "vbll":"D-VBLL", "vbll_mean":"D-VBLL mean only"}
CLASSES=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    result=read(RUN/"test_results.json")
    PUBLIC.mkdir(parents=True,exist_ok=True)
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "savefig.dpi":180,"figure.constrained_layout.use":True})
    recomputed=[]
    for seed in [11,22,33]:
        folder=RUN/f"seed_{seed}"
        z=np.load(folder/"test_predictions.npz")
        for method in NAMES:
            m=metrics(z[method],z["labels"])
            old=next(r for r in result["per_seed"] if r["seed"]==seed and r["method"]==method)
            for k in ["accuracy","nll","brier","ece"]:
                if abs(m[k]-old[k])>1e-10:
                    raise AssertionError(f"Metric mismatch: {seed}/{method}/{k}")
            if m["clipped_true_probability_count"]:
                raise AssertionError("NLL clipping occurred; qualify/report before release")
            recomputed.append(dict(seed=seed,method=method,**{k:m[k] for k in ["accuracy","nll","brier","ece"]}))
        shutil.copy2(folder/"test_predictions.npz",PUBLIC/f"seed_{seed}_predictions.npz")
        for f in ["selection.json","cnn_history.json","head_history.json"]:
            shutil.copy2(folder/f,PUBLIC/f"seed_{seed}_{f}")
    with (PUBLIC/"metrics_per_seed.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(recomputed[0]));w.writeheader();w.writerows(recomputed)
    for file in ["test_results.json","environment.json","data_manifest.json","freeze_manifest.json","run_fingerprint.json"]:
        shutil.copy2(RUN/file,PUBLIC/file)
    # Preselected seed 11, not the best-performing seed.
    z=np.load(RUN/"seed_11"/"test_predictions.npz")
    fig,ax=plt.subplots(1,2,figsize=(9.5,3.7))
    for method,color in [("map","#5470a3"),("vbll","#c45535"),("map_temperature","#29927e")]:
        r=metrics(z[method],z["labels"])
        bins=[b for b in r["reliability"] if b["count"]>=10]
        ax[0].plot([b["confidence"] for b in bins],[b["accuracy"] for b in bins],"o-",markersize=4,
                   label=f"{NAMES[method]} (ECE {r['ece']:.2f}%)",color=color)
        ax[1].hist(z[method].max(1),bins=np.linspace(0,1,16),histtype="step",linewidth=1.8,
                   label=NAMES[method],color=color)
    ax[0].plot([0,1],[0,1],"--",color="gray",label="Perfect calibration")
    ax[0].set(xlabel="Mean confidence within bin",ylabel="Fraction correct",xlim=(.2,1),ylim=(.2,1),title="Reliability: fixed seed 11")
    ax[0].legend(fontsize=7.5,loc="upper left")
    ax[1].set(xlabel="Maximum predicted probability",ylabel="Number of test images",title="Confidence distribution")
    ax[1].legend(fontsize=8)
    fig.savefig(FIG/"reliability.png");plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(9.5,3.3))
    for seed,color in [(11,"#5470a3"),(22,"#c45535"),(33,"#29927e")]:
        h=read(RUN/f"seed_{seed}"/"cnn_history.json")
        ax[0].plot([r["epoch"] for r in h],[r["training_ce"] for r in h],label=f"Seed {seed}",color=color)
        ax[1].plot([r["epoch"] for r in h],[r["validation_nll"] for r in h],label=f"Seed {seed}",color=color)
    ax[0].set(title="CNN training cross-entropy",xlabel="Epoch",ylabel="Mean loss")
    ax[1].set(title="CNN validation NLL",xlabel="Epoch",ylabel="NLL (natural log)")
    for a in ax:a.legend(fontsize=8);a.grid(alpha=.2)
    fig.savefig(FIG/"learning_curves.png");plt.close(fig)
    pred=z["vbll"].argmax(1); y=z["labels"]
    cm=np.zeros((10,10),dtype=int)
    np.add.at(cm,(y,pred),1)
    fig,ax=plt.subplots(figsize=(7.8,6.2))
    im=ax.imshow(cm/10,cmap="Blues",vmin=0,vmax=100)
    ax.set_xticks(range(10),CLASSES,rotation=45,ha="right",fontsize=8)
    ax.set_yticks(range(10),CLASSES,fontsize=8)
    for i in range(10):
        for j in range(10):
            val=cm[i,j]/10
            if val>=.5:ax.text(j,i,f"{val:.1f}",ha="center",va="center",fontsize=7,color="white" if val>50 else "#23334a")
    ax.set(xlabel="Predicted class",ylabel="True class",title="D-VBLL confusion matrix: seed 11 (% within true class)")
    fig.colorbar(im,ax=ax,fraction=.035,pad=.02,label="Percent")
    fig.savefig(FIG/"confusion.png");plt.close(fig)
    off=cm.copy();np.fill_diagonal(off,0)
    indices=np.argsort(off.ravel())[::-1][:5]
    confusions=[dict(true=CLASSES[i//10],predicted=CLASSES[i%10],count=int(off.ravel()[i])) for i in indices]
    summary=dict(metrics_recomputed=True,nll_clipped_count=0,seed11_largest_confusions=confusions,
        seed11_class_accuracy={c:float(cm[i,i]/10) for i,c in enumerate(CLASSES)},
        total_training_seconds=sum(read(RUN/f"seed_{s}"/"selection.json")["total_seconds"] for s in [11,22,33]))
    (PUBLIC/"analysis_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(result["aggregate"],indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
