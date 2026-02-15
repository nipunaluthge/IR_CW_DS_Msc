# IR Coursework - Run Instructions (Local)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Key runs (Parts 4–6)
```bash
python src/06b_build_mapping_goal_filtered_v2.py
python evaluation/eval_metrics_multiK.py
streamlit run app/app.py
```

## Outputs to verify
- outputs/mapping_topk_goal_filtered_v2.csv
- outputs/overall_metrics.json
- outputs/strategy_metrics.csv
- outputs/eval_per_strategy.csv
