import wandb
from neural_numpy.builder import NetworkBuilder
from neural_numpy.loss import CategoricalCrossEntropy
from neural_numpy.optimizer import ADAM, SGD
from data import DataLoader

# will probably yield very shitty results without running on the full dataset

sweep_configuration = {
    "method": "bayes",
    "metric": {
        "name": "val_acc",
        "goal": "maximize",
    },
    "parameters": {
        "epochs": {"value": 10},  # Fixed value
        "batch_size": {"values": [32, 64, 128]},
        "learning_rate": {
            "max": 0.1,
            "min": 0.0001,
            "distribution": "log_uniform_values",
        },
        "hidden_layers": {"values": [1, 2, 3, 4]},
        "hidden_units": {"values": [64, 128, 256, 512]},
        "activation": {"values": ["ReLU", "Sigmoid", "Tanh"]},
        "weight_initializer": {"values": ["Xavier", "He", "RandomNormal"]},
        "optimizer": {"values": ["adam", "sgd"]},
        "weight_decay": {"values": [0.0, 1e-3, 1e-4]},
    },
}

X_train, y_train, X_test, y_test = DataLoader.load_cifar10(
    normalize=True, flatten=True, one_hot=True
)

subset_size = 5000
X_train = X_train[:subset_size]
y_train = y_train[:subset_size]

val_split = 0.2
split_idx = int(X_train.shape[0] * (1 - val_split))
X_val = X_train[split_idx:]
y_val = y_train[split_idx:]
X_train_split = X_train[:split_idx]
y_train_split = y_train[:split_idx]

input_dim = X_train.shape[1]
num_classes = y_train.shape[1]


def train_sweep():
    with wandb.init() as run:
        config = wandb.config

        builder = NetworkBuilder()
        network = builder.build_from_wandb(
            input_size=input_dim, output_size=num_classes, config=config
        )

        if config.optimizer.lower() == "adam":
            optimizer = ADAM(
                learning_rate=config.learning_rate,
                weight_decay=getattr(config, "weight_decay", 0.0),
            )
        elif config.optimizer.lower() == "sgd":
            optimizer = SGD(
                learning_rate=config.learning_rate,
                weight_decay=getattr(config, "weight_decay", 0.0),
            )

        loss_fn = CategoricalCrossEntropy()

        network.train(
            X=X_train_split,
            y=y_train_split,
            X_val=X_val,
            y_val=y_val,
            loss_function=loss_fn,
            epochs=config.epochs,
            optimizer=optimizer,
            batch_size=config.batch_size,
        )


def main():
    wandb.login()

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="neural-numpy-sweep")

    wandb.agent(sweep_id, function=train_sweep, count=20)


if __name__ == "__main__":
    main()
