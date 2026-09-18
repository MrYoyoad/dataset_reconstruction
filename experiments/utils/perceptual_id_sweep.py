#!/usr/bin/env python3
"""Perceptual-identification sweep over the saved tensors of every reconstruction cell on disk.

Walks the four sources of experiments/utils/perceptual_id.py (oracle_ladder, ntk_vs_cert, bootstrap_chart,
decoder_chart), scores every arm of every saved cell, and writes
  results/perceptual_id/<source>_<jobid>.jsonl    one row per (cell, arm, target, image)
  results/perceptual_id/summary_<jobid>.md        per (cell, arm, target): exact landed / SSIM top-1 / top-5 / feature top-1
                                                  / median SSIM to truth and to control, plus the letters-ladder table
                                                  that answers "at which chart error is the recovery still identified
                                                  top-1 among 100 while exact landings are already 0"
  results/perceptual_id/summary_<jobid>.json      the same summary, machine-readable, with the manifest of files seen
The summary records which sources had no files (absent) and which LSF jobs of those families were still running when the
sweep ran, so it can be rerun later with the same command.

  python -u -m experiments.utils.perceptual_id_sweep --sources oracle_ladder --include mlp_letter_a_eps0.pth mlp_letter_a_pca.pth   # smoke
  python -u -m experiments.utils.perceptual_id_sweep                                                                                 # everything on disk
"""
import argparse, fnmatch, glob, json, os, socket, subprocess, sys, time
import numpy as np
import torch

from experiments.utils.perceptual_id import SOURCES, load_arms, score_arm, summarise, lineup_figure, LAND, N_DECOYS, DECOY_SEED, log

JOB_FAMILIES = {"oracle_ladder": "ol_", "ntk_vs_cert": "nvc_", "bootstrap_chart": "bsc_", "decoder_chart": "dc_"}   # LSF job-name prefixes of the producers


def running_jobs():
    try:
        out = subprocess.run(["bjobs", "-w", "-noheader"], capture_output=True, text=True, timeout=30).stdout
        return [ln.split()[6] for ln in out.splitlines() if len(ln.split()) > 6]
    except Exception as e:                                   # bjobs missing on the node: record that, do not fail the sweep
        return [f"bjobs unavailable: {e}"]


def md_table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]


def fmt(v, nd=3):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=list(SOURCES))
    ap.add_argument("--include", nargs="*", default=None, help="basename globs; only matching .pth files are scored")
    ap.add_argument("--exclude", nargs="*", default=["clf_*.pth"], help="basename globs to skip (default: the bootstrap classifier caches)")
    ap.add_argument("--out-dir", default="results/perceptual_id"); ap.add_argument("--fig-dir", default="figures/perceptual_id")
    ap.add_argument("--figures", action="store_true", help="write a line-up grid per attacker-output arm")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True); job = os.environ.get("LSB_JOBID", "local"); tag = (a.tag + "_" if a.tag else "")
    t_start = time.time(); running = running_jobs()
    log(f"# perceptual_id_sweep  job={job} host={socket.gethostname()} sources={a.sources} include={a.include} landing<{LAND} lineup=1+{N_DECOYS} decoy_seed={DECOY_SEED}")
    manifest = {}; summary = []
    for src in a.sources:
        pattern, _ = SOURCES[src]
        files = sorted(glob.glob(pattern))
        if a.include: files = [f for f in files if any(fnmatch.fnmatch(os.path.basename(f), g) for g in a.include)]
        files = [f for f in files if not any(fnmatch.fnmatch(os.path.basename(f), g) for g in a.exclude)]
        still = sorted({j for j in running if j.startswith(JOB_FAMILIES[src])})
        manifest[src] = dict(pattern=pattern, n_files=len(files), files=[os.path.basename(f) for f in files], absent=(len(files) == 0), producer_jobs_running=still)
        log(f"\n#### {src}: {len(files)} file(s){' -- ABSENT' if not files else ''}; producer jobs still running: {len(still)} {still[:6]}{'...' if len(still) > 6 else ''}")
        out_path = os.path.join(a.out_dir, f"{tag}{src}_{job}.jsonl")
        for f in files:
            try: arms = load_arms(f, src)
            except Exception as e:
                log(f"#   ERROR loading {f}: {type(e).__name__}: {e}"); manifest[src].setdefault("errors", []).append(f"{os.path.basename(f)}: {e}"); continue
            for arm in arms:
                t0 = time.time()
                try: rows = score_arm(arm)
                except Exception as e:
                    log(f"#   ERROR scoring {arm['cell']} {arm['arm']}: {type(e).__name__}: {e}"); manifest[src].setdefault("errors", []).append(f"{arm['cell']}/{arm['arm']}: {e}"); continue
                sm = summarise(rows)
                head = dict(source=src, cell=arm["cell"], arm=arm["arm"], target=arm["target"], attacker_output=arm["attacker_output"], ckpt=arm.get("ckpt"), meta=arm["meta"])
                with open(out_path, "a") as fh:
                    for r in rows: fh.write(json.dumps(dict(head, **r, job=job, path=arm["path"])) + "\n")
                summary.append(dict(head, **sm))
                log(f"  {arm['cell']:<70s} {arm['arm']:<26s} vs {arm['target']:<7s} landed {sm['landed']}/{sm['n']}  SSIM top1 {sm['top1_ssim']} top5 {sm['top5_ssim']}  "
                    f"L2 top1 {sm['top1_l2']}  feat top1 {sm['top1_feat']} top5 {sm['top5_feat']}  SSIM truth {sm['ssim_truth_median']:.3f} ctrl {sm['ssim_control_median']:.3f}  [{time.time()-t0:.1f}s]")
                if a.figures and arm["attacker_output"]:
                    lineup_figure(arm, rows, os.path.join(a.fig_dir, f"{src}_{arm['cell']}_{arm['arm']}_{arm['target']}.png"))

    # ---------------------------------------------------------------- summary tables
    L = [f"# Perceptual identification sweep -- job {job}, {time.strftime('%Y-%m-%d %H:%M')}", "",
         f"Tier 1 = exact landing (relative pixel error < {LAND}). Tier 2 = line-up identification: the truth among 1+{N_DECOYS} public images of the same class "
         f"(train split, fixed decoy seed {DECOY_SEED}); rank of the truth by SSIM / pixel L2 / base-model feature L2 to the recovery. "
         "'ctrl' = SSIM(recovery, nearest public image to the truth). Rows with attacker_output=False are references (chart projections, oracle selections), not attacks.", "",
         "## Sources seen", ""]
    for src, m in manifest.items():
        L.append(f"- **{src}**: {m['n_files']} file(s)" + (" -- ABSENT, rerun later with the same command" if m["absent"] else "") +
                 (f"; producer jobs still running at sweep time: {', '.join(m['producer_jobs_running'])}" if m["producer_jobs_running"] else "; no producer job running") +
                 (f"; errors: {len(m['errors'])}" if m.get("errors") else ""))
    L += ["", "## Per cell (attacker outputs first)", ""]
    hdr = ["source", "cell", "arm", "target", "attack?", "n", "exact landed", "SSIM top-1", "SSIM top-5", "L2 top-1", "feat top-1", "feat top-5", "med SSIM truth", "med SSIM ctrl", "chart err"]
    def chart_err(s):
        m = s["meta"]; return m.get("proj_err_mean", m.get("chart_repr_err_median", m.get("chart_err_median")))
    rows_ = [[s["source"], s["cell"], s["arm"], s["target"], "yes" if s["attacker_output"] else "ref", s["n"], s["landed"], s["top1_ssim"], s["top5_ssim"], s["top1_l2"], s["top1_feat"], s["top5_feat"],
              fmt(s["ssim_truth_median"]), fmt(s["ssim_control_median"]), fmt(chart_err(s), 4)]
             for s in sorted(summary, key=lambda s: (not s["attacker_output"], s["source"], s["cell"], s["arm"], s["target"]))]
    L += md_table(hdr, rows_)

    # the letters ladder (and every ladder example): identification vs measured chart error, attacker arm only
    lad = [s for s in summary if s["source"] == "oracle_ladder" and s["arm"] == "cert_best"]
    if lad:
        L += ["", "## Oracle ladder: identification as the chart error grows (cert_best vs raw truth)", "",
              "eps is the ladder's perturbation; 'chart err' is the MEASURED mean projection error of the true images. Exact landings die first; the question is where SSIM top-1 dies.", ""]
        for ex in sorted({s["meta"]["example"] for s in lad}):
            L += [f"### {ex}", ""]
            rows_ex = sorted([s for s in lad if s["meta"]["example"] == ex], key=lambda s: (s["meta"]["wrong_release"], float("inf") if s["meta"]["proj_err_mean"] is None else s["meta"]["proj_err_mean"]))
            L += md_table(["cell", "chart", "eps", "chart err (mean)", "exact landed", "SSIM top-1", "SSIM top-5", "L2 top-1", "feat top-1", "med SSIM truth", "med SSIM ctrl", "med SSIM truth-vs-ctrl"],
                          [[s["cell"], s["meta"]["chart"] + (" WRONG-RELEASE" if s["meta"]["wrong_release"] else ""), fmt(s["meta"]["eps"], 3), fmt(s["meta"]["proj_err_mean"], 4), s["landed"], s["top1_ssim"], s["top5_ssim"],
                            s["top1_l2"], s["top1_feat"], fmt(s["ssim_truth_median"]), fmt(s["ssim_control_median"]), fmt(s["ssim_truth_vs_control_median"])] for s in rows_ex])
            ok = [s for s in rows_ex if not s["meta"]["wrong_release"] and s["meta"]["chart"] == "oracle"]
            zero_land = [s for s in ok if s["landed"] == 0]; ident = [s for s in zero_land if s["top1_ssim"] == s["n"]]
            any_ident = [s for s in zero_land if s["top1_ssim"] > 0]
            if zero_land:
                L += ["", f"- first oracle cell with 0 exact landings: eps {fmt(zero_land[0]['meta']['eps'],3)} (chart err {fmt(zero_land[0]['meta']['proj_err_mean'],4)}); "
                          f"SSIM top-1 there {zero_land[0]['top1_ssim']}/{zero_land[0]['n']}"]
                if ident: L.append(f"- 0 exact landings but ALL {ident[0]['n']} SSIM-identified top-1 among 100: eps up to {fmt(ident[-1]['meta']['eps'],3)} (chart err {fmt(ident[-1]['meta']['proj_err_mean'],4)})")
                if any_ident: L.append(f"- 0 exact landings but at least one SSIM top-1: eps up to {fmt(any_ident[-1]['meta']['eps'],3)} (chart err {fmt(any_ident[-1]['meta']['proj_err_mean'],4)}, {any_ident[-1]['top1_ssim']}/{any_ident[-1]['n']})")
            L.append("")
    md = "\n".join(L) + f"\n\n_{len(summary)} arms scored in {time.time()-t_start:.0f}s; rows in {a.out_dir}/{tag}<source>_{job}.jsonl_\n"
    open(os.path.join(a.out_dir, f"{tag}summary_{job}.md"), "w").write(md)
    json.dump(dict(job=job, cmd=" ".join(sys.argv), manifest=manifest, summary=summary), open(os.path.join(a.out_dir, f"{tag}summary_{job}.json"), "w"), indent=1, default=str)
    log("\n" + md)


if __name__ == "__main__":
    main()
