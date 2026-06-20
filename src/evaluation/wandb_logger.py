import wandb


def start_run(run_name: str, config_dict: dict):
    wandb.init(
        project="rfp-rag-evaluation",
        name=run_name,
        config=config_dict,
    )


def log_metrics(metrics: dict):
    wandb.log(metrics)


def finish_run():
    wandb.finish()