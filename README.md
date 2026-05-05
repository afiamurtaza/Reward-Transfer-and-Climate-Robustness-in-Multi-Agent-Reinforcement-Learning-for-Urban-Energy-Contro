# Reward Transfer and Climate Robustness in Multi-Agent Reinforcement Learning for Urban Energy Control

**Afia Murtaza · Maleeha Fatima · Salam Orabi**

A multi-agent reinforcement learning (MARL) study that investigates zero-retraining climate transfer from a temperate US residential district to Dubai desert conditions using the [CityLearn 2023](https://github.com/intelligent-environments-lab/CityLearn) six-building benchmark. We compare IPPO and MAPPO across three reward designs and introduce the **Reward Transfer Score (RTS)** to measure reward–KPI alignment under distributional shift.

---

## Research Questions

| # | Question |
|---|----------|
| RQ1 | Does climate transfer degrade KPI performance uniformly across all metrics? |
| RQ2 | Which reward design is most robust under zero-shot climate transfer? |
| RQ3 | Does IPPO or MAPPO generalise better to the Dubai climate? |
| RQ4 | Does climate transfer alter the structure of the cost–discomfort trade-off frontier? |

---

## Experimental Design

| Dimension | Options |
|-----------|---------|
| **Algorithm** | IPPO (decentralised), MAPPO (centralised critic) |
| **Reward design** | Flat shared · Local individual · UAE-weighted |
| **Training steps** | 250,000 |
| **Random seeds** | 42 · 2 · 7 |
| **Source climate** | CityLearn Challenge 2023 Phase 3 (US temperate) |
| **Transfer climate** | Dubai TMYx real hourly weather data |

Full training configuration (learning rate, rollout length, PPO clip range, etc.) is documented in the report: [`Adv_AI_Project.pdf`](Adv_AI_Project.pdf).

---

## Key Results

- **Cost and carbon emissions decrease** under Dubai transfer while **thermal discomfort increases** — policies reduce energy consumption but cannot adapt to intensified cooling demand.
- **RTS ≈ 0** (−0.0226, *p* = 0.9662): reward change is statistically decoupled from KPI change under transfer. Reward alone is not a reliable deployment signal.
- **MAPPO** preserves the cost–discomfort Pareto geometry ~3.16× better than IPPO.
- **IPPO with local individual reward** achieves the most balanced overall KPI profile and lowest training variance.
- Seasonal analysis identifies four distinct regimes: winter degradation, spring coordination advantage, summer saturation, and autumn partial generalisation.

---

## Repository Structure

```
.
├── CityLearn_MARL_Setup.ipynb                  # Environment setup & architecture definition
├── CityLearn_MARL_IPPO_MAPPO_Baseline.ipynb    # Fixed baseline — corrected IPPO & MAPPO
├── MARL_Energy_Control_Training.ipynb          # Full 2×3 training matrix — configurable seed
├── CityLearn_UAE_Evaluation.ipynb              # Zero-shot Dubai transfer evaluation
├── Single_Seed_Analysis.ipynb                  # Single-seed KPI & transfer analysis
├── Multi_Seed_Summary.ipynb                    # Multi-seed aggregation, figures & tables
│
├── models/
│   ├── ippo_flat_shared_250k[_seed2][_seed3].pt
│   ├── ippo_local_individual_250k[_seed2][_seed3].pt
│   ├── ippo_uae_weighted_250k[_seed2][_seed3].pt
│   ├── mappo_flat_shared_250k[_seed2][_seed3].pt
│   ├── mappo_local_individual_250k[_seed2][_seed3].pt
│   └── mappo_uae_weighted_250k[_seed2][_seed3].pt
│
├── citylearn_dubai/                            # Full-year Dubai CityLearn dataset (6 buildings)
├── seasonal_dubai_citylearn/                   # UAE seasonal datasets
│   ├── citylearn_dubai_Q1_winter/
│   ├── citylearn_dubai_Q2_spring/
│   ├── citylearn_dubai_Q3_summer/
│   └── citylearn_dubai_Q4_autumn/
├── dubai_weather.csv                           # Raw Dubai TMYx weather data
│
├── results_dubai_evaluation/                   # UAE transfer KPIs & reward CSVs + figures
├── results_analysis_single_seed/               # RTS, Pareto, seasonal analysis outputs
├── results_analysis_multiseed/                 # Mean ± std KPI tables & figures across seeds
│
└── Adv_AI_Project.pdf                          # Full project report
```

---

## File & Folder Reference

### Notebooks

| File | Purpose |
|------|---------|
| [`CityLearn_MARL_Setup.ipynb`](CityLearn_MARL_Setup.ipynb) | Sets up the CityLearn 2023 environment, defines observation/action spaces and the IPPO/MAPPO architectures. |
| [`CityLearn_MARL_IPPO_MAPPO_Baseline.ipynb`](CityLearn_MARL_IPPO_MAPPO_Baseline.ipynb) | Fixed baseline run - validates the corrected IPPO and MAPPO implementations and reward pipeline. |
| [`MARL_Energy_Control_Training.ipynb`](MARL_Energy_Control_Training.ipynb) | Full 2 × 3 training matrix (algorithm × reward design); set `SEED` and `RUN_INDEX` to reproduce seeds 42, 2, 7. |
| [`CityLearn_UAE_Evaluation.ipynb`](CityLearn_UAE_Evaluation.ipynb) | Loads trained checkpoints and runs zero-shot evaluation on the Dubai climate dataset. |
| [`Single_Seed_Analysis.ipynb`](Single_Seed_Analysis.ipynb) | Per-seed KPI and transfer analysis - produces per-condition figures. |
| [`Multi_Seed_Summary.ipynb`](Multi_Seed_Summary.ipynb) | Aggregates across seeds; produces final paper figures, tables, and the RTS analysis. |

### Data

| Path | Purpose |
|------|---------|
| [`citylearn_dubai/`](citylearn_dubai/) | Full-year Dubai CityLearn dataset (6 buildings + schema, pricing, carbon, weather). |
| [`seasonal_dubai_citylearn/`](seasonal_dubai_citylearn/) | Dubai dataset split into four seasonal regimes for the seasonal analysis. |
| [`seasonal_dubai_citylearn/citylearn_dubai_Q1_winter/`](seasonal_dubai_citylearn/citylearn_dubai_Q1_winter/) | Jan–Mar Dubai weather (12–32 °C). |
| [`seasonal_dubai_citylearn/citylearn_dubai_Q2_spring/`](seasonal_dubai_citylearn/citylearn_dubai_Q2_spring/) | Apr–Jun Dubai weather (22–42 °C). |
| [`seasonal_dubai_citylearn/citylearn_dubai_Q3_summer/`](seasonal_dubai_citylearn/citylearn_dubai_Q3_summer/) | Jul–Sep Dubai weather (30–47 °C). |
| [`seasonal_dubai_citylearn/citylearn_dubai_Q4_autumn/`](seasonal_dubai_citylearn/citylearn_dubai_Q4_autumn/) | Oct–Dec Dubai weather (20–38 °C). |
| [`dubai_weather.csv`](dubai_weather.csv) | Raw Dubai TMYx hourly weather source file used to build the CityLearn datasets. |

### Models

| Path | Purpose |
|------|---------|
| [`models/`](models/) | All trained PyTorch checkpoints - 2 algorithms × 3 reward designs × 3 seeds (18 files, 250k steps each). |

### Results

| Path | Purpose |
|------|---------|
| [`results_dubai_evaluation/`](results_dubai_evaluation/) | UAE transfer KPIs and reward CSVs from the zero-shot Dubai evaluation, plus US-vs-UAE comparison figures. |
| [`results_analysis_single_seed/`](results_analysis_single_seed/) | RTS, Pareto-frontier, and seasonal analysis outputs -> RQ1–RQ4 figures and tables for the primary seed. |
| [`results_analysis_multiseed/`](results_analysis_multiseed/) | Mean ± std KPI tables and figures aggregated across seeds 42, 2, 7. |

### Report

| File | Purpose |
|------|---------|
| [`Adv_AI_Project.pdf`](Adv_AI_Project.pdf) | Full write-up: methodology, training configuration, results, and discussion. |

---

## Notebook Execution Order

Run notebooks in the following order to reproduce results from scratch:

```
1. CityLearn_MARL_Setup.ipynb
sets up environment, defines observation/action spaces
      ↓  
2. CityLearn_MARL_IPPO_MAPPO_Baseline.ipynb
trains fixed baseline, validates reward functions and pipeline
      ↓  
3. MARL_Energy_Control_Training.ipynb   (set SEED and RUN_INDEX per seed)
trains all 6 conditions for one seed; repeat for seeds 42, 2, 7
      ↓  
4. CityLearn_UAE_Evaluation.ipynb
loads trained models, runs zero-shot Dubai evaluation
      ↓  
5. Single_Seed_Analysis.ipynb
produces per-condition KPI and transfer figures
      ↓ 
6. Multi_Seed_Summary.ipynb
aggregates across seeds, produces final paper figures & tables 

```

To skip training, pre-trained 250k model checkpoints are provided in `models/`.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install citylearn torch numpy pandas matplotlib
```

> Tested with CityLearn 2.1.2, PyTorch 2.0, Python 3.9.

---

## UAE Climate Data

The `seasonal_dubai_citylearn/` directory contains real TMYx hourly weather profiles (outdoor temperature, relative humidity, solar irradiance) for Dubai, split into four seasonal regimes. All building parameters (thermal loads, HVAC sizing, battery capacity, pricing) are held constant from the US training dataset — only the weather file is replaced, isolating the effect of climate shift.

| Directory | Season | Dubai Temp. Range |
|-----------|--------|-------------------|
| `seasonal_dubai_citylearn/citylearn_dubai_Q1_winter/` | Jan–Mar | 12–32 °C |
| `seasonal_dubai_citylearn/citylearn_dubai_Q2_spring/` | Apr–Jun | 22–42 °C |
| `seasonal_dubai_citylearn/citylearn_dubai_Q3_summer/` | Jul–Sep | 30–47 °C |
| `seasonal_dubai_citylearn/citylearn_dubai_Q4_autumn/` | Oct–Dec | 20–38 °C |

---

## Report

The full write-up including methodology, results tables, and figures is available in [`Adv_AI_Project.pdf`](Adv_AI_Project.pdf).
