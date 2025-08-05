from jobflow import Flow, Maker
from pp.flows.prepare_dataset import GenerateDFTData
from pp.jobs.train import *
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pymatgen.core import Structure

__all__ = [
    'TrainFlowMaker'
]


@dataclass
class TrainFlowMaker(Maker):
    """
    A Maker to train a NN with DeepH-E3

    Args:
        generate_dft_kwargs: dict
            kwargs for GenerateDFTData, see flows/prepare_dataset
            for details

    """
    name = 'Train Flow'
    generate_dft_kwargs: dict[str,Any]
    update_default_training_kwargs: dict[str,str]
    save_dir: Optional[Union[str, Path]] = None
    run_generate_flow: bool = True
    training_python_script: str = './deephe3-train.py'
    train_options: Optional[str] = None

    def make(
            self,
            structure: Optional[Structure]=None
        ) -> Flow:
        jobs = []

        if self.run_generate_flow:
            generate_flow = GenerateDFTData(**self.generate_dft_kwargs).make(structure=structure)
            jobs.append(generate_flow)

        train_ini = write_train_ini(
            update_default_training_kwargs = self.update_default_training_kwargs,
            save_dir = self.save_dir
        )
        train_job = run_training(
            train_ini = train_ini,
            training_python_script = self.training_python_script,
            options = self.train_options,
            prev_deps = generate_flow.output if self.run_generate_flow else None
        )
        jobs.append(train_job)

        return Flow(jobs, output = [j.output for j in jobs], name=self.name)




