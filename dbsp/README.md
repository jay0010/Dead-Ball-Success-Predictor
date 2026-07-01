# Dead Ball Success Predictor  (v2)

A prescriptive **set-piece / dead-ball analytics** web app for **first-team
coaches and sports analysts**, built for UCD module **MIS41420 – Sports &
Performance Analytics** (Group Assessment, 40%).

For any planned routine it instantly predicts:

1. **Expected Goals (xG)** of the chance — *Engine A (regressor)*
2. **Second-ball retention probability** — *Engine B (classifier)*

…and gives a **detailed second-ball plan** plus charts explaining *why*.

### What's new in v2
- **Eight dead-ball situations:** corner, free kick (wide), free kick (central),
  **penalty, throw-in, goal kick, kick-off, play restart**.
- **New look:** a clean light **Green & Blue** theme (re-skin via
  `.streamlit/config.toml`).
- **Branded opening page** — the app opens on a themed welcome page; click
  **“Launch the predictor”** to enter the tool. A **“Home”** button (sidebar)
  returns to it. Inside the tool, **“Roll new scenario”** loads a fresh routine.
- **More tactical triggers:** runner from deep, far-post overload, designated
  second-wave runner, quick delivery, dummy run (plus the original four).
- **Detailed second-ball section:** likely drop zone, expected 2nd-phase xG,
  active second-ball levers, and concrete coach recommendations.

---

## How to run — 3 simple steps

> ⚠️ A Streamlit app is **not** started with `python3 dbsp.py`. Use
> `streamlit run dbsp.py`. Also keep the whole folder together — `dbsp.py` needs
> the `src/`, `models/` and `data/` folders next to it.

Open a terminal **inside this folder** and run:

```bash
# Step 1 — install the libraries (first time only)
pip install -r requirements.txt

# Step 2 — train the two models (first time only, ~1 minute)
python src/train_models.py

# Step 3 — launch the app
streamlit run dbsp.py
```

Your browser opens automatically at **http://localhost:8501**.

### Quick sanity check before Step 3
Run `ls` (Mac/Linux) or `dir` (Windows). You should see
`dbsp.py  src  models  data  requirements.txt`. If you only see `dbsp.py`,
you are in the wrong folder — `cd` into the unzipped
`dead-ball-success-predictor` folder first. (The zip already ships trained
models, so you can usually skip straight to Step 3.)

### Using the app
- The app opens on a welcome page — click **“Launch the predictor”** to enter.
- The **Control Panel** on the left sets the situation, delivery, defence,
  personnel and tactical triggers.
- The cards on the right update instantly: **xG**, **second-ball retention %**
  and an overall **routine rating**.
- The **Second-ball retention plan** explains where the loose ball drops, which
  of your levers are active, and what to add to keep possession.
- Click **Roll new scenario** any time to jump to a fresh routine.

---

## Real StatsBomb data (optional)

```bash
pip install statsbombpy
PREFER_REAL=1 python src/train_models.py        # macOS / Linux
# Windows PowerShell:
#   $env:PREFER_REAL=1 ; python src/train_models.py
```
If the download fails it falls back to synthetic data automatically.

## Project structure

```
dead-ball-success-predictor/
├── dbsp.py                       # Streamlit web app
├── requirements.txt
├── README.md
├── .streamlit/config.toml       # light Green & Blue theme
├── src/
│   ├── data_ingestion.py        # 8 dead-ball types + triggers (StatsBomb / synthetic)
│   ├── feature_engineering.py   # numeric feature matrix
│   ├── scenario.py              # random scenarios + second-ball logic
│   └── train_models.py          # trains Engine A & Engine B
├── models/                      # trained models + metrics (created by training)
└── data/                        # generated dataset
```

*Lead Analyst: Jayan Kokru & Project Team — MIS41420, UCD College of Business.*
