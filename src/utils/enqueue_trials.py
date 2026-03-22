"""Enqueue specific hyperparameter combinations into an existing Optuna study."""
import optuna

study_name = "slr_opt_config=configs-pruning_sweep-l0_fixed_v3"
storage_db = f"sqlite:////home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/sqlite_dbs/{study_name}.db"

study = optuna.load_study(study_name=study_name, storage=storage_db)

# Trial 3 params (the one that peaked at 0.842)
study.enqueue_trial({
    "skel_learning_rate": 0.00048072468767305465,
    "bert_hidden_layers": 6,
    "bert_hidden_dim": 384,
    "bert_att_heads": 8,
    "bert_intermediate_size": 2048,
    "l0_end_weight": 10.0,
    "classifier_dropout": 0.1,
})

# Trial 4 params
study.enqueue_trial({
    "skel_learning_rate": 9.414793904681351e-05,
    "bert_hidden_layers": 4,
    "bert_hidden_dim": 256,
    "bert_att_heads": 4,
    "bert_intermediate_size": 1536,
    "l0_end_weight": 10.0,
    "classifier_dropout": 0.1,
})

print("Enqueued 2 trials successfully.")
print(f"Total trials in study: {len(study.trials)}")
