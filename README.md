# Reward Design Transferability Under Climate Shift: Evaluating MARL Energy Controllers in Dubai Desert Weather

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
├── CityLearn_MARL_Setup.ipynb                 # Environment setup & architecture definition
├── CityLearn_MARL_IPPO_MAPPO_Training.ipynb  # Fixed baseline — corrected IPPO & MAPPO
├── MARL_Energy_Control_Training.ipynb               # Full 2×3 training matrix — configurable seed
├── CityLearn_UAE_Evaluation.ipynb             # Zero-shot Dubai transfer evaluation
├── Single_Seed_Analysis.ipynb                 # Single-seed KPI & transfer analysis
├── Multi_Seed_Summary.ipynb                   # Multi-seed aggregation, figures & tables
│
├── models/
│   ├── ippo_flat_shared_250k[_seed2][_seed3].pt
│   ├── ippo_local_individual_250k[_seed2][_seed3].pt
│   ├── ippo_uae_weighted_250k[_seed2][_seed3].pt
│   ├── mappo_flat_shared_250k[_seed2][_seed3].pt
│   ├── mappo_local_individual_250k[_seed2][_seed3].pt
│   └── mappo_uae_weighted_250k[_seed2][_seed3].pt
│
├── citylearn_uae_Q1_winter/                   # UAE seasonal datasets (6 buildings each)
├── citylearn_uae_Q2_spring/
├── citylearn_uae_Q3_summer/
├── citylearn_uae_Q4_autumn/
├── citylearn_uae_dubai/                       # Full-year Dubai dataset
├── dubai_weather.csv                          # Raw Dubai TMYx weather data
│
├── results_uae 7/                             # UAE transfer KPIs & reward CSVs + figures
├── results_analysis 8/                        # RTS, Pareto, seasonal analysis outputs
├── results_multiseed 9/                       # Mean ± std KPI tables & figures across seeds
│
└── Adv_AI_Project.pdf                         # Full project report
```

---

## Notebook Execution Order

Run notebooks in the following order to reproduce results from scratch:

```
1. CityLearn_MARL_Setup.ipynb
      ↓  sets up environment, defines observation/action spaces
2. CityLearn_MARL_IPPO_MAPPO_Training.ipynb
      ↓  trains fixed baseline, validates reward functions and pipeline
3. MARL_Energy_Control_Training.ipynb   (set SEED and RUN_INDEX per seed)
      ↓  trains all 6 conditions for one seed; repeat for seeds 42, 2, 7
4. CityLearn_UAE_Evaluation.ipynb
      ↓  loads trained models, runs zero-shot Dubai evaluation
5. Single_Seed_Analysis.ipynb
      ↓  produces per-condition KPI and transfer figures
6. Multi_Seed_Summary.ipynb
      ↓  aggregates across seeds, produces final paper figures & tables
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

The `citylearn_uae_*` directories contain real TMYx hourly weather profiles (outdoor temperature, relative humidity, solar irradiance) for Dubai, split into four seasonal regimes. All building parameters (thermal loads, HVAC sizing, battery capacity, pricing) are held constant from the US training dataset — only the weather file is replaced, isolating the effect of climate shift.

| Directory | Season | Dubai Temp. Range |
|-----------|--------|-------------------|
| `citylearn_uae_Q1_winter/` | Jan–Mar | 12–32 °C |
| `citylearn_uae_Q2_spring/` | Apr–Jun | 22–42 °C |
| `citylearn_uae_Q3_summer/` | Jul–Sep | 30–47 °C |
| `citylearn_uae_Q4_autumn/` | Oct–Dec | 20–38 °C |

---

## Report

The full write-up including methodology, results tables, and figures is available in [`Adv_AI_Project.pdf`](Adv_AI_Project.pdf).
