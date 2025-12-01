from typing import Dict, List, Any
from enum import Enum
from .network import NeuralNetwork
from .layer import Dense
from .activation import ReLU, Sigmoid, Tanh, Softmax
from .initializers import Xavier, He, RandomNormal
from rich.console import Console
from rich.table import Table
from rich import box


class ActivationType(str, Enum):
    """Available activation functions."""

    RELU = "ReLU"
    SIGMOID = "Sigmoid"
    TANH = "Tanh"
    SOFTMAX = "Softmax"


class InitializerType(str, Enum):
    """Available weight initializers."""

    XAVIER = "Xavier"
    HE = "He"
    RANDOM_NORMAL = "RandomNormal"


class NetworkBuilder:
    """
    Factory class for constructing NeuralNetwork instances.
    """

    def __init__(self):
        self.console = Console()

    def build_mlp(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: int,
        hidden_units: int,
        activation: ActivationType = ActivationType.RELU,
        output_activation: ActivationType = ActivationType.SOFTMAX,
        weight_initializer: InitializerType = InitializerType.XAVIER,
        bias_initializer: InitializerType = InitializerType.XAVIER,
    ) -> NeuralNetwork:
        network = NeuralNetwork()
        w_init = self._get_initializer(weight_initializer)
        b_init = self._get_initializer(bias_initializer)

        # 1. Input -> First Hidden Layer
        current_input = input_size

        # 2. Hidden Layers
        for _ in range(hidden_layers):
            network.add_layer(
                Dense(
                    input_size=current_input,
                    output_size=hidden_units,
                    weight_initializer=w_init,
                    bias_initializer=b_init,
                )
            )
            network.add_layer(self._get_activation(activation))
            current_input = hidden_units  # Output of this layer is input to next

        # 3. Output Layer
        network.add_layer(
            Dense(
                input_size=current_input,
                output_size=output_size,
                weight_initializer=w_init,
                bias_initializer=b_init,
            )
        )
        network.add_layer(self._get_activation(output_activation))

        self._print_summary(network)
        return network

    def build_from_config(self, config: List[Dict[str, Any]]) -> NeuralNetwork:
        """
        Constructs a NeuralNetwork from a generic list of layer configurations.
        Retained for flexibility if non-MLP architectures are needed.
        """
        network = NeuralNetwork()

        for layer_conf in config:
            layer_type = layer_conf.get("type")

            if layer_type == "Dense":
                output_size = layer_conf.get("output_size")
                input_size = layer_conf.get("input_size")  # Optional

                w_init_name = layer_conf.get("weight_initializer", "Xavier")
                b_init_name = layer_conf.get("bias_initializer", "Xavier")
                w_init = self._get_initializer(w_init_name)
                b_init = self._get_initializer(b_init_name)

                layer = Dense(
                    output_size=output_size,
                    input_size=input_size,
                    weight_initializer=w_init,
                    bias_initializer=b_init,
                )
                network.add_layer(layer)

            elif layer_type in [a.value for a in ActivationType]:
                # Convert string to Enum and get instance
                network.add_layer(self._get_activation(ActivationType(layer_type)))
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

        self._print_summary(network)
        return network

    def _get_initializer(self, name: str | InitializerType):
        # Handle both string (from generic config) and Enum
        name_str = name.value if isinstance(name, InitializerType) else name

        if name_str == InitializerType.XAVIER.value:
            return Xavier()
        elif name_str == InitializerType.HE.value:
            return He()
        elif name_str == InitializerType.RANDOM_NORMAL.value:
            return RandomNormal()
        return Xavier()

    def _get_activation(self, activation_type: ActivationType):
        if activation_type == ActivationType.RELU:
            return ReLU()
        elif activation_type == ActivationType.SIGMOID:
            return Sigmoid()
        elif activation_type == ActivationType.TANH:
            return Tanh()
        elif activation_type == ActivationType.SOFTMAX:
            return Softmax()
        raise ValueError(f"Unsupported activation: {activation_type}")

    def _print_summary(self, network: NeuralNetwork):
        """Prints a rich table summary of the built network."""
        table = Table(title="Neural Network Architecture", box=box.ROUNDED)

        table.add_column("Layer ID", justify="center", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta")
        table.add_column("Input Shape", justify="right", style="green")
        table.add_column("Output Shape", justify="right", style="green")
        table.add_column("Params", justify="right", style="yellow")

        total_params = 0

        for i, layer in enumerate(network.get_layers()):
            layer_name = layer.__class__.__name__

            params_count = 0
            for param in layer.get_parameters():
                params_count += param.data.size

            total_params += params_count

            if isinstance(layer, Dense):
                in_shape = str(layer.input_size) if layer.input_size else "?"
                out_shape = str(layer.output_size)
            else:
                in_shape = "-"
                out_shape = "-"

            table.add_row(
                str(i + 1), layer_name, in_shape, out_shape, f"{params_count:,}"
            )

        table.add_section()
        table.add_row(
            "Total", "", "", "", f"[bold yellow]{total_params:,}[/bold yellow]"
        )

        self.console.print(table)
