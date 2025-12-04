import os
import wandb
import main  

# Run "python -m neural_numpy.sweep_config" to run a sweep

sweep_config = {
    "method": "bayes",  # "random" or "bayes"
    "metric": {
        "name": "val_acc",  
        "goal": "maximize",
    },
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 0.001,
            "max": 0.01,
        },
        "hidden_layers": {
            "values": [2, 3, 4,5,6],
        },
        "hidden_units": {
            "values": [64, 128, 256, 512, 1024],
        },
        "weight_decay": {
            "values": [0.0, 1e-4, 1e-3],
        },
        "epochs": {
            "value": 20,
        },
    },
}

def train():
    main.main()


if __name__ == "__main__":
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="skraldespanden",
    )

    wandb.agent(sweep_id, function=train, count=2)