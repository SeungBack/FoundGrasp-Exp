"""
Thin experiment logger supporting comet_ml and wandb backends.
Usage:
    from utils.logger import logger
    logger.init(backend='comet', project='ViTGrasp', name='run1', config={}, log_dir='/tmp')
    logger.log({'loss': 0.1}, step=100)
    logger.finish()
"""


class ExperimentLogger:
    def __init__(self):
        self._backend = None
        self._experiment = None

    def init(self, backend, project, name, config_dict, log_dir, mode='online', entity=None):
        self._backend = backend if mode != 'disabled' else None
        if self._backend is None:
            return

        if backend == 'comet':
            from comet_ml import Experiment
            self._experiment = Experiment(
                project_name=project,
                auto_metric_logging=False,
                auto_param_logging=False,
                auto_output_logging=False,
            )
            self._experiment.set_name(name)
            self._experiment.log_parameters(config_dict)

        elif backend == 'wandb':
            import wandb
            wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config_dict,
                dir=log_dir,
                mode=mode,
            )

    def log(self, metrics, step=None):
        if self._backend is None:
            return
        if self._backend == 'comet':
            self._experiment.log_metrics(metrics, step=step)
        elif self._backend == 'wandb':
            import wandb
            wandb.log(metrics, step=step) if step is not None else wandb.log(metrics)

    def finish(self):
        if self._backend is None:
            return
        if self._backend == 'comet':
            self._experiment.end()
        elif self._backend == 'wandb':
            import wandb
            wandb.finish()


logger = ExperimentLogger()
