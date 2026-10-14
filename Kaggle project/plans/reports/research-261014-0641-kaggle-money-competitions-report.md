# Research: Kaggle cash-prize comps feasible solo (Claude + free GPU + RTX 4050 6GB laptop)

Date: 2026-10-14. Source: live kaggle.com/competitions (Active, sort by prize) + each overview/rules page, read via in-app browser.
Laptop: Ryzen 7 7435HS, 24GB RAM, RTX 4050 6GB. Kaggle CLI not installed, no kaggle.json.

## Reality check
- 2026 Kaggle = mostly hackathons/code comps. Classic tabular/CV "train model, climb LB" cash comps: none active. Playground S6E10 = swag only.
- Code comps w/ big pools have 3k-6k teams, deadline short, need heavy GPU. Solo top-3 w/ free GPU = lottery.
- Hackathons (writeup judged by humans) = few teams, many payouts, GPU irrelevant. Claude helps most here.

## Active comps with money (as of 2026-10-14)

| Comp | Prize | Teams | Deadline | Type | Fit |
|---|---|---|---|---|---|
| NFL Big Data Bowl 2027 | $100k (12 payouts + $10k grand) | 222 | **2027-01-06** | Hackathon writeup, tracking data analytics | **BEST** |
| Gemma 4 Dev Agent - Paper Track | $35k (15k best paper, 10k best resource, 10k best application) | 210 | **2026-11-12** | Paper/writeup | GOOD (tight) |
| Gemma 4 Dev Agent (main) | $65k (37k/18k/10k) | 2698 | 2026-12-02 (entry 11-25) | SWE-agent, submit prompts/tools/skills/LoRA for Gemma-4-31B | MAYBE, hard to eval locally |
| ARC Prize 2026 Paper Track | $450k pool (3 awards 50k/20k/5k + 375k bonus if >4.5/5) | 252 | 2026-11-09 | Paper, but "Accuracy" criterion needs LB submission | SKIP |
| ARC-AGI-3 / ARC-AGI-2 | $850k / $700k | 4260 / 2854 | 2026-11-02 | Code comp, interactive/reasoning | SKIP (19 days, SOTA labs) |
| RSNA Knee Abnormality | $77k | 5951 | ~9 days | Medical imaging | SKIP |
| Enveda CASMI Molecule ID | $50k | 3233 | ~2 months | Mass spec -> structure, chem ML | SKIP (domain + compute) |
| Playground S6E10 Airline Satisfaction | swag | 1764 | 18 days | Tabular | Practice only |
| Pokemon PTCG Playground | swag | 353 | 3 months | Simulation | Practice only |

## #1 NFL Big Data Bowl 2027 (recommended)
- Task: link 10Hz Combine sensor tracking -> NFL regular-season performance. Writeup <=2000 words, <10 tables/figures, public Kaggle notebook. Must use tracking data.
- Judges (NFL analytics staff): Football 30%, Data Science 30%, Writeup 20%, Viz 20%. Rewards narrow, novel, actionable metric on one position/drill, NOT "evaluate everyone".
- Payouts: Open Track $45k (3x$10k finalists, 3x$5k runners-up). University Track $45k same split — only groups/individuals entirely undergrads (verification may be required). Finalists invited to Combine; +$10k grand prize.
- 6 paid slots per track vs 222 teams now (past BDBs reach several hundred). Odds ~ top 3-5% of entrants — best in list. If undergrad (UIT?) -> enter University Track, much smaller pool.
- Eligibility: worldwide except OFAC/Crimea/Cuba/Iran/NK -> Vietnam OK. Winner pays own taxes; need tax docs.
- Fit: runs fine on laptop/Kaggle CPU (pandas, polars, sklearn, xgboost, plots). No GPU needed. Claude = EDA, metric design, stats, viz, writing.
- Risk: need football domain knowledge; time ~3 months (plenty). Differentiator = clean causal/statistical rigor + 1 clear metric + great figures.

## #2 Gemma 4 Developer Agent - Paper Track
- Deadline 2026-11-12 (4 weeks). Prizes: Best Paper $15k, Best New Resource (dataset/tool) $10k, Best New Application $10k. Judged Novelty/Quality/Relevance/Verifiability/Clarity. Highlighted at NeurIPS 2026 expo.
- Only 210 teams. Resource/Application categories are achievable w/o GPU: e.g. code-graph tool/dataset built from the provided graph+embedding data, repo-navigation skill library.
- Risk: need "original, unpublished" research; tight 4 weeks; quality bar of Google judges.
- Main comp: eval is Gemma-4-31B-it QAT w4a16 only; 12h patch budget per run. Submission may be prompts/tools/skills only (no LoRA) -> Claude-friendly in principle, but you can't run 31B on 6GB; iterating needs Kaggle's quota. 2698 teams. Treat as stretch; try only if paper track work produces a harness.

## Skip rationale
- ARC: LB accuracy dominates; labs + 4k teams; 19 days. Paper track needs LB score.
- RSNA/Enveda: domain heavy, large data, many GPU-hrs; 3-6k teams.

## Recommended plan
1. Now: join NFL BDB 2027 (+ Gemma Paper Track). Download data, EDA on Combine tracking. Pick 1 position group + 1 drill + 1 outcome (e.g. DB change-of-direction -> coverage separation).
2. Oct 14-Nov 12: parallel Gemma paper-track writeup only if a concrete resource idea emerges in first 3 days; else go all-in on NFL.
3. Nov-Dec: iterate NFL writeup, validate stats (out-of-sample by draft year), polish figures, submit before Jan 6 (aim Dec 28).
4. Keep Playground comps as free practice for pipeline; no cash.
5. Set up: `pip install kaggle`, create API token (user action), for notebooks/data pulls.

## Unresolved
- Are you undergrad (UIT)? Decides University Track eligibility; need proof.
- NFL rules on AI-assisted writing/code: not found in rules text scanned; check Rules tab fully before relying on Claude-generated prose.
- Gemma paper-track "original unpublished" + dataset availability (graph/embedding) not inspected.
- Payout to Vietnam: tax forms/bank transfer details unknown.
- Search snippets mentioned Biohub Cell Tracking ($60k), AI Agent Security ($50k), Pokemon PTCG Strategy ($240k) — absent from the live Active list; likely ended/not yet open. Check "Upcoming"/new launches weekly.
- Kaggle "Active" list showed 2 pages; page 2 not read (only knowledge/getting-started likely).

## Addendum 2026-10-14: payout + tax (Vietnam)
- Rules: no payout method stated. Old Kaggle rules: "check or wire transfer". Kaggle forum: payout takes 3-30 days after ALL docs returned; veteran says ~6 weeks, sometimes more. NFL rules: ~30 days after docs.
- Docs: W-8BEN for non-US. Forum: national ID may substitute for ITIN (anecdote, 1 user, Israel).
- US-Vietnam income tax treaty signed 2015, NOT in force (secondary sources; IRS list not directly verified). Treasury "working towards" renegotiation (June 2026). => no treaty rate; default 30% US withholding on prize to nonresident likely (not verified for Kaggle specifically).
- Possible: prize sourced abroad (work done in VN) => maybe no US withholding; unverified, ask tax adviser.
- Vietnam PIT on foreign prize income: not researched.
- Unresolved: actual Kaggle payout channel today; actual withholding for VN winners; refund via 1040-NR feasibility; VN tax treatment.
