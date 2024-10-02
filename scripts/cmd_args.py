# Copyright 2023 Kévin Garanger
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from pathlib import Path

from core import (ActivationType, ArchitectureType, DataScalingMethodType,
                  DatasetSampler, DatasetSamplerType, FeatureType,
                  InitializerType, LossType, MetricType, OptimizerType,
                  SchedulerType)
from TFENN.core import SymmetricTensorNotationType, TensorSymmetryClassType


class CmdArgs:
    @staticmethod
    def add_config_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--jax_config", type=str, nargs="*", help="JAX config flags", default=[]
        )
        return parser

    @staticmethod
    def add_model_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "-i",
            "--input_shape",
            type=int,
            required=True,
            nargs="+",
            help="Shape of the model inputs",
        )
        parser.add_argument(
            "-o",
            "--output_shape",
            type=int,
            required=True,
            nargs="+",
            help="Shape of the model outputs",
        )
        parser.add_argument(
            "-n",
            "--layer_size",
            type=int,
            choices=range(1, 1000),
            metavar="[1-1000]",
            default=20,
            help="Number of neurons per hidden layer",
        )
        parser.add_argument(
            "-m",
            "--layer_count",
            type=int,
            choices=range(1, 100),
            metavar="[1-100]",
            default=3,
            help="Number of hidden layers",
        )
        parser.add_argument(
            "-f",
            "--feature_type",
            type=FeatureType,
            choices=FeatureType,
            default=FeatureType.SCALAR,
            help="Type of network features",
        )
        parser.add_argument(
            "-k",
            "--sym_cls",
            type=TensorSymmetryClassType,
            choices=TensorSymmetryClassType,
            help="Symmetry class type to use for TFENNs",
            default=TensorSymmetryClassType.NONE,
        )
        parser.add_argument(
            "-t",
            "--architecture_type",
            type=ArchitectureType,
            choices=ArchitectureType,
            default=ArchitectureType.MLP,
            help="Type of network architecture to use",
        )
        parser.add_argument(
            "-a",
            "--activation",
            type=ActivationType,
            choices=ActivationType,
            default=ActivationType.RELU,
            help="Activation function to use",
        )
        parser.add_argument(
            "--output_activation",
            type=ActivationType,
            choices=ActivationType,
            help="Activation function to use for the output layer",
        )
        parser.add_argument(
            "-r",
            "--use_io_rotator",
            action="store_true",
            help="Add a learnable matrix to rotate input/output pairs",
        )
        parser.add_argument(
            "--kernel_initializer",
            type=InitializerType,
            choices=InitializerType,
            default=InitializerType.LE_CUN_NORMAL,
            help="Kernel initializer to use",
        )
        parser.add_argument(
            "--bias_initializer",
            type=InitializerType,
            choices=InitializerType,
            default=InitializerType.ZERO,
            help="Bias initializer to use",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=0,
            help="Random seed to generate PRNG key",
        )

        return parser

    @staticmethod
    def add_dataset_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "-d",
            "--dataset_files",
            type=Path,
            nargs="+",
            required=True,
            help="Path to the dataset files",
        )
        parser.add_argument(
            "--val_dataset_files",
            type=Path,
            nargs="*",
            help="Path to the validation dataset files",
        )
        parser.add_argument(
            "--data_type",
            type=FeatureType,
            choices=FeatureType,
            default=FeatureType.SCALAR,
            help="Type of data inputs and outputs",
        )
        parser.add_argument(
            "--data_notation",
            type=SymmetricTensorNotationType,
            choices=SymmetricTensorNotationType,
            help="Notation of the symmetric tensors",
        )
        parser.add_argument(
            "--data_tensor_dim",
            type=int,
            default=2,
            help="Dimension of the symmetric tensors",
        )
        parser.add_argument(
            "--file_in_cols",
            type=int,
            nargs="+",
            help="Column indices to use as input features",
        )
        parser.add_argument(
            "--file_out_cols",
            type=int,
            nargs="+",
            help="Column indices to use as output features",
        )
        parser.add_argument(
            "--sequence_length",
            type=int,
            help="Length of the sequences in the dataset for sequential data",
        )
        parser.add_argument(
            "--max_dataset_size",
            type=int,
            help="Maximum number of samples to use from the dataset",
        )
        parser.add_argument(
            "--dataset_split",
            type=float,
            help=(
                "Fraction of provided dataset to dedicate to training "
                "(the rest is used for validation)"
            ),
        )
        parser.add_argument(
            "--dataset_scaling",
            type=DataScalingMethodType,
            choices=DataScalingMethodType,
            default=DataScalingMethodType.NORMAL,
            help="Method to use for scaling the dataset",
        )
        parser.add_argument(
            "--dataset_scaling_file",
            type=Path,
            help="Path to the dataset scaling file for manual scaling",
        )
        parser.add_argument(
            "--scale_per_feature",
            action="store_true",
            help="Whether to scale each feature separately",
        )
        parser.add_argument(
            "--scale_per_step",
            action="store_true",
            help="Whether to scale each step separately for sequential data",
        )
        parser.add_argument(
            "--dataset_sampler",
            type=DatasetSamplerType,
            choices=DatasetSamplerType,
            default=DatasetSamplerType.RANDOM_SHUFFLE,
            help="Type of dataset sampler to use",
        )
        parser.add_argument(
            "--grad_dataset_files",
            type=Path,
            nargs="+",
            help="Path to the gradient dataset file. If not, the main dataset file will be used",
        )
        parser.add_argument(
            "--grad_file_out_cols",
            type=int,
            nargs="+",
            help="Column indices to use as output gradient",
        )

        return parser

    @staticmethod
    def add_training_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--loss",
            type=LossType,
            choices=LossType,
            default=LossType.SE,
            help="Loss function to use",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
            help="Batch size for training",
        )
        parser.add_argument(
            "--optimizer",
            type=OptimizerType,
            choices=OptimizerType,
            default=OptimizerType.ADAMW,
            help="Optimizer to use",
        )
        parser.add_argument(
            "--optimizer_options",
            type=str or float,
            nargs="+",
            default=[],
            help="Parameters for the optimizer as a list of positional and keyword arguments",
        )
        parser.add_argument(
            "--scheduler",
            type=SchedulerType,
            choices=SchedulerType,
            default=SchedulerType.NONE,
            help="Scheduler to use",
        )
        parser.add_argument(
            "--scheduler_options",
            type=str or float,
            nargs="+",
            default=[],
            help="Parameters for the scheduler as a list of positional and keyword arguments",
        )
        parser.add_argument(
            "--max_epochs",
            type=int,
            default=None,
            help="Maximum number of epochs to train for",
        )
        parser.add_argument(
            "--max_time",
            type=int,
            default=None,
            help="Maximum time (in minutes) to train for",
        )
        parser.add_argument(
            "--checkpoint_dir",
            type=Path,
            help="Checkpoint of the model to start training from",
        )
        parser.add_argument(
            "--reinit_opt",
            action="store_true",
            help="Whether to reinitialize the optimizer when loading a checkpoint",
        )
        parser.add_argument(
            "--freeze_layers",
            type=int,
            default=0,
            help="Number of layers to freeze during training",
        )
        parser.add_argument(
            "--gradient_clip",
            type=float,
            help="Clip gradients during training",
        )

        return parser

    @staticmethod
    def add_save_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "-s",
            "--save_dir",
            type=Path,
            required=True,
            help="Directory to save the model and training results",
        )
        parser.add_argument(
            "--save_interval",
            type=int,
            default=10,
            help="Number of epochs between saving the model",
        )

        return parser

    @staticmethod
    def add_logging_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--logging_level",
            type=str,
            default="WARNING",
            help="Logging level",
        )
        parser.add_argument(
            "--print_epoch_interval",
            type=int,
            default=1,
            help="Number of epochs between printing training results",
        )
        parser.add_argument(
            "--print_iteration_interval",
            type=int,
            default=100,
            help="Number of iterations between printing epoch progress",
        )
        parser.add_argument(
            "--print_metrics",
            type=MetricType,
            nargs="+",
            default=[],
            help="Metrics to print during training",
        )

        return parser

    @staticmethod
    def add_testing_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--prediction_files",
            type=Path,
            help="Path to the predictions file",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=1,
            help="Batch size for testing",
        )
        parser.add_argument(
            "--loss",
            type=LossType,
            choices=LossType,
            default=LossType.SE,
            help="Loss function to use",
        )
        parser.add_argument(                                                                                                
            "--checkpoint_dir",                                                                                             
            type=Path,                                                                                                      
            help="Checkpoint of the model to start training from",                                                          
        )  
        return parser
