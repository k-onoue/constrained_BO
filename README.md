# to do

- debug utils

## memo

- https://arxiv.org/pdf/2210.10199 をベースのアルゴリズムにしようと思う
- https://github.com/facebookresearch/bo_pr
- ちょっと深層化可能か怪しいので確認する

/constrained_BO/results/2024-08-09/optuna/Warcraft

```
-> % tree results
results
├── 2024-07-25
│   └── Warcraft
│       └── exhaustive_search.json
└── 2024-08-09
    └── optuna
        └── Warcraft
            ├── best_trial_gp.json
            ├── best_trial_nsga.json
            ├── best_trial_random.json
            ├── best_trial_tpe.json
            ├── study_gp.pkl
            ├── study_nsga.pkl
            ├── study_random.pkl
            └── study_tpe.pkl

6 directories, 9 files
```