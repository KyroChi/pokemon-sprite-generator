import datetime
import os

from accelerate import notebook_launcher, Accelerator
from tqdm.auto import tqdm

class ExperimentConfig(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return str(self.__dict__)


class Experiment(object):
    def __init__(self, 
                 experiment_name,
                 project_dir,
                 num_epochs,
                 save_model_epochs,
                 mixed_precision='fp16',
                 gradient_accumulation_steps=1,
                 log_with='tensorboard',
                 **kwargs):
        self.args = None
        
        self.accelerator = None

        if experiment_name is None:
            raise ValueError("'experiment_name' is required.")
        self.experiment_name = experiment_name

        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if project_dir is None:
            raise ValueError("'project_dir' is required.")
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)

        self.project_dir = os.path.join(project_dir, experiment_name, now)
        os.makedirs(self.project_dir)

        self.config = ExperimentConfig(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with=log_with,
            num_epochs=num_epochs,
            save_model_epochs=save_model_epochs,
            **kwargs
        )

        self.config.project_dir = self.project_dir 
        self.train_dataloader = None

    def train_setup(self):
        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with=self.config.log_with,
            project_dir=self.config.project_dir
        )

        if self.accelerator.is_main_process:
            if self.config.project_dir is not None:
                os.makedirs(self.config.project_dir, exist_ok=True)
            self.accelerator.init_trackers("train_examples")

    def train_evaluation(self, *args, **kwargs):
        raise NotImplementedError("Evaluation not implemented")
    
    def train_epoch_start(self, *args, **kwargs):
        pass

    def train_epoch_end(self, *args, **kwargs):
        pass

    def train_single_iteration(idx, batch, self, *args, **kwargs):
        raise NotImplementedError("train_single_iteration method must be implemented")

    def train_teardown(self):
        self.accelerator.end_training()

    def _train_loop(self, *args, **kwargs):
        self.train_setup()
        global_step = 0
        for epoch in range(self.config.num_epochs):
            num_batches = self.config.__dict__.get('num_batches', None)
            if num_batches is None:
                num_batches = len(self.train_dataloader)
            progress_bar = tqdm(total=num_batches, disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            dataiter = iter(self.train_dataloader)
            batches_seen = 0

            for idx, batch in enumerate(dataiter):
                logging = self.train_single_iteration(idx, batch, epoch, *args, **kwargs)

                logging['global_step'] = global_step
                self.accelerator.log(logging, step=global_step)

                progress_bar.update(1)
                progress_bar.set_postfix(**logging)
                global_step += 1

                batches_seen += 1
                if batches_seen >= num_batches:
                    break

            if self.accelerator.is_main_process:
                try:
                    self.train_evaluation(epoch, *args, **kwargs)
                except NotImplementedError:
                    print("Warning: Evaluation method is not implemented.")
                    
    def run(self, **kwargs):
        notebook_launcher(self._train_loop, self.args, num_processes=kwargs.get("num_processes", 1))