import sys

import numpy as np

sys.setrecursionlimit(2500)

# ------------------------------ Sisyphus -------------------------------------

import sisyphus.toolkit as tk
from ukp.huggingface.search import *
from ukp.huggingface.training import *
from ukp.huggingface.task_arithmetic import MakeAndApplyTaskVectorsJob, \
 MakeTaskVectorsJob, MakeAndApplyTaskVectorsCapeJob, MakeAndApplyTaskVectorsCapeWithFisherJob, MakeAndApplyTaskVectorsEWRJob
from ukp.huggingface.evaluation import *
from i6_core.text import PipelineJob


Path = tk.Path

code_root = gs.CODE_ROOT

def train_model(method, model_name_or_path, dataset, dataset_config_name, model_description, per_device_train_batch_size=4, gradient_accumulation_steps=8,
                dataset_train_split="train", dataset_val_split="validation", num_epochs=10, time_rqmt=24, mem_rqmt=24, gpu_mem=10, learning_rate=6.25e-5,
                per_device_eval_batch_size=8): #junling modify
    config = {
        'model_name_or_path': model_name_or_path,
        'predict_with_generate': True,
        'method': method,
        'learning_rate': learning_rate,
        'per_device_train_batch_size': per_device_train_batch_size,
        # 'per_device_eval_batch_size': per_device_eval_batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'cache_dir': gs.CACHE_DIR,
    }
    train_data_config = {
        'dataset_name': os.path.join(code_root, f'dialog/datasets/{dataset}.py'),
        'dataset_config_name': dataset_config_name,
        'dataset_train_split': dataset_train_split,
        'dataset_val_split': dataset_val_split,
    }

    train_job = HuggingfaceTrainingJob(
        code_root=code_root,
        config=config,
        train_data_config=train_data_config,
        num_epochs=num_epochs,
        mem_rqmt=mem_rqmt,
        time_rqmt=time_rqmt,
        gpumem=gpu_mem
    )
    train_job.add_alias(f"cleaned_repo/train_job-{dataset}_{dataset_config_name}_{model_description}")
    tk.register_output(f'{dataset}_{dataset_config_name}_{model_description}', train_job.out_best_model)
    return train_job

def evaluate_model(method, model_name_or_path, dataset, dataset_config_name, model_description, per_device_eval_batch_size=8,
                   dataset_test_split="test", time_rqmt=2, mem_rqmt=24, gpu_mem=10, calculate_q2=False, generation_beam_size=None):
    config = {
        'model_name_or_path': model_name_or_path,
        'predict_with_generate': True,
        'method': method,
        'per_device_eval_batch_size': per_device_eval_batch_size,
    }
    search_data_config = {
        'dataset_name': os.path.join(code_root, f'dialog/datasets/{dataset}.py'),
        'dataset_config_name': dataset_config_name,
        'dataset_test_split': dataset_test_split,
    }

    if generation_beam_size is not None:
        config["generation_beam_size"] = generation_beam_size

    search_job = HuggingfaceSearchJob(
        code_root=code_root,
        model_path=model_name_or_path,
        config=config,
        search_data_config=search_data_config,
        mem_rqmt=mem_rqmt,
        time_rqmt=time_rqmt,
        gpumem=gpu_mem,
    )

    if dataset == "dstc11":
        scoring_job = CalculateMetricsForMultiDocJob(
            code_root,
            search_data_config["dataset_name"],
            search_data_config['dataset_test_split'],
            search_job.out_search_file,
            time_rqmt=2,
        )
    else:
        scoring_job = CalculateMetricsJob(
            code_root,
            search_data_config["dataset_name"],
            search_data_config['dataset_test_split'],
            search_job.out_search_file,
            time_rqmt=2
        )
    tk.register_output(f'results/{dataset}/{dataset_config_name}_{method}_{model_description}.metrics.json', scoring_job.out_results_file)

def calculate_fisher_information(method, model_name_or_path, dataset, dataset_config_name, model_description, per_device_eval_batch_size=8,
                   dataset_test_split="validation", time_rqmt=2, mem_rqmt=24, gpu_mem=10):
    assert "fisher" in method
    config = {
        'model_name_or_path': model_name_or_path,
        'predict_with_generate': True,
        'method': method,
        'per_device_eval_batch_size': per_device_eval_batch_size,
        'track_fim': True,
    }
    search_data_config = {
        'dataset_name': os.path.join(code_root, f'dialog/datasets/{dataset}.py'),
        'dataset_config_name': dataset_config_name,
        'dataset_test_split': dataset_test_split,
    }

    search_job = HuggingfaceSearchJob(
        code_root=code_root,
        model_path=model_name_or_path,
        config=config,
        search_data_config=search_data_config,
        mem_rqmt=mem_rqmt,
        time_rqmt=time_rqmt,
        gpumem=gpu_mem
    )
    tk.register_output(f'results/{dataset}/{dataset_config_name}_{method}_{model_description}_fisher', search_job.out_checkpoints_dir)

    return search_job.out_checkpoints_dir


import asyncio


async def run_models(**config):
    baseline_model = train_model(
        config["method"],
        config["model_name_or_path"],
        config["train_dataset"],
        config["train_dataset_config_name"],
        config["model_description"],
        gpu_mem=config["gpu_mem_train"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        dataset_train_split=config["dataset_train_split"],
        dataset_val_split=config["dataset_val_split"],
        num_epochs=config["num_epochs"],
        time_rqmt=config["train_time_rqmt"]
    )

    for test_dataset in config["test_datasets"]:
        # Evaluate baseline model
        evaluate_model(
            config["method"],
            baseline_model,
            test_dataset,
            config["test_dataset_config_name"],
            config["model_description"] + "_task_arithmetic",
            gpu_mem=config["gpu_mem_test"],
            dataset_test_split=config["dataset_test_split"],
            calculate_q2=True
        )

        # Calculate Fisher information for the baseline model
        fisher_base = calculate_fisher_information(
            config["fisher_estimation_method"],
            baseline_model,
            config["train_dataset"],
            config["train_dataset_config_name"],
            config["model_description"],
            dataset_test_split=config["dataset_val_split"] if not config["dataset_val_split"] == config["dataset_test_split"] else config["dataset_train_split"],
            time_rqmt=4,
            gpu_mem=config["gpu_mem_fisher"]
        )

        # EWR
        for scaling_factor in [0.15]:
            scaling_factor = round(scaling_factor, 3)

            new_model = MakeAndApplyTaskVectorsEWRJob(
                config["code_root"],
                baseline_model,
                [],
                fisher_base,
                [],
                scaling_factors_experts=[],
                scaling_factors_anti_experts=[scaling_factor]
            )

            evaluate_model(
                config["method"],
                new_model,
                test_dataset,
                config["test_dataset_config_name"],
                config["model_description"] + "_task_arithmetic_ewr_" + str(scaling_factor),
                gpu_mem=config["gpu_mem_test"],
                dataset_test_split=config["dataset_test_split"],
                calculate_q2=scaling_factor == 0.15
            )

async def task_arithmetic():
    config = {
        "method": "document_grounded_generation",
        "model_name_or_path": "google/flan-t5-base",
        "model_description": "flan_t5_dialogue_inpainting",
        "train_dataset": "wow",
        "test_datasets": ["wow"],
        "train_dataset_config_name": "response_generation",
        "test_dataset_config_name": "response_generation",
        "expert_dataset_name": "wow",
        "anti_expert_dataset_name": "faithdial",
        "expert_dataset_config_name": "cape_expert",
        "anti_expert_dataset_config_name": "hallucinated_response",
        "dataset_train_split": "train",
        "dataset_val_split": "validation",
        "dataset_test_split": "test",
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "per_device_eval_batch_size": 8,
        "gpu_mem_train": 12,
        "gpu_mem_test": 10,
        "num_epochs": 10,
        "num_expert_epochs": 5,
        "gpu_mem_fisher": 12
    }

    # run_models(**config)

async def async_main():
    await task_arithmetic()

async def py():
    await async_main()

if __name__ == "__main__":
    asyncio.run(async_main())
