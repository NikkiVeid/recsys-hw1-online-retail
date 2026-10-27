"""Prespecified validation search; test is evaluated after every selection is frozen."""
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from models import ItemKNN, IALS, EASE, FM, CustomIALS
from evaluation import evaluate

SEED = 42
PRIMARY_METRIC = 'NDCG@10'
CLASSES = {'ItemKNN': ItemKNN, 'iALS': IALS, 'EASE': EASE, 'FM': FM,
           'Custom iALS': CustomIALS, 'FM IDs only': FM}
# A compact budget fixed before looking at either holdout.
SEARCH = {
    'ItemKNN': [dict(k=k, shrink=s) for k in (30, 100) for s in (0, 20)],
    'iALS': [dict(factors=32, reg=r, alpha=a, iterations=12, seed=SEED)
             for r in (1., 10.) for a in (5., 20.)],
    'EASE': [dict(reg=r) for r in (10., 100., 500.)],
    'FM': [dict(factors=32, lr=.005, weight_decay=1e-5, epochs=e,
                negatives=3, batch_size=4096, metadata=True, seed=SEED) for e in (4, 8)],
}


def search_models(data, truth, output_dir='results'):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trials, chosen = [], {}

    def search_one(name, configs):
        best_score, best_config = -np.inf, None
        for params in configs:
            start = time.perf_counter()
            model = CLASSES[name](**params).fit(data)
            train_time = time.perf_counter() - start
            metrics, _ = evaluate(model, data, truth)
            trials.append({'model': name, 'params': json.dumps(params, sort_keys=True),
                           'fit_seconds': train_time, **metrics})
            print(f'{name}: NDCG@10={metrics[PRIMARY_METRIC]:.5f}, Recall@10={metrics["Recall@10"]:.5f}', flush=True)
            if metrics[PRIMARY_METRIC] > best_score:
                best_score, best_config = metrics[PRIMARY_METRIC], params.copy()
            del model
            gc.collect()
        chosen[name] = best_config
        pd.DataFrame(trials).to_csv(output/'validation_trials.csv', index=False)

    for name, configs in SEARCH.items():
        search_one(name, configs)
    # Controlled target ablation: same ALS confidence, rank, regularization, seed and iterations.
    search_one('Custom iALS', [dict(**chosen['iALS'], tau_days=t, saturation=2.) for t in (90., 180., 365.)])
    # Controlled metadata ablation: identical optimizer and chosen training budget.
    chosen['FM IDs only'] = dict(chosen['FM'], metadata=False)
    (output/'selected_params.json').write_text(json.dumps(chosen, indent=2), encoding='utf-8')
    return pd.DataFrame(trials), chosen


def final_evaluation(data, truth, chosen, output_dir='results'):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, per_user, examples = [], {}, {}
    # chosen is already frozen and persisted. No test-driven adaptation is performed.
    for name, params in chosen.items():
        start = time.perf_counter()
        model = CLASSES[name](**params).fit(data)
        train_time = time.perf_counter() - start
        start = time.perf_counter()
        metrics, detailed = evaluate(model, data, truth)
        rows.append({'model': name, **metrics, 'fit_seconds': train_time,
                     'eval_seconds': time.perf_counter()-start, 'params': json.dumps(params, sort_keys=True)})
        per_user[name] = detailed
        detailed.to_csv(output/(name.replace(' ', '_')+'_per_user.csv'))
        example_users = np.array(sorted(truth)[:3])
        examples[name] = model.recommend(example_users, data.binary).tolist()
        print(f'Test {name}: NDCG@10={metrics[PRIMARY_METRIC]:.5f}, Recall@10={metrics["Recall@10"]:.5f}', flush=True)
        del model
        gc.collect()
    result = pd.DataFrame(rows).sort_values(PRIMARY_METRIC, ascending=False).reset_index(drop=True)
    result.to_csv(output/'test_results.csv', index=False)
    (output/'recommendation_examples.json').write_text(json.dumps(examples, indent=2))
    return result, per_user, examples
