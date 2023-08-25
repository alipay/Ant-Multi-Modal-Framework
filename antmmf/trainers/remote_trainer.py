import json
import yaml
from antmmf.trainers.base_trainer import check_configuration

_SOLUTION = "solution_122_330"
_SCENE_TYPE = "custom_module"
_SOURCE = "launcher_antmmf"

_MODULE_NAME = "custom_module"
_MODEL_NAME = "custom_model"

_DEFAULT_SCENE = "antmmf_test"

_DEFAULT_WORKER_NUM = 1
_DEFAULT_WORKER_CORE = 32
_DEFAULT_WORKER_MEMORY = 1024 * 32
_DEFAULT_WORKER_GPU_COUNT = 2


class RemoteTrainer:
    def __init__(self, config):
        self.config = check_configuration(config)
        self.packages = None
        self.code = None
        self.datasets = None
        self.scene = _DEFAULT_SCENE
        self.user = None
        self.server = None
        self.hpo = None

        from launcher.cloud.common import WorkerSpec

        self.worker_spec = WorkerSpec(
            num_workers=_DEFAULT_WORKER_NUM,
            cpu=_DEFAULT_WORKER_CORE,
            memory=_DEFAULT_WORKER_MEMORY,
            gpu=_DEFAULT_WORKER_GPU_COUNT,
        )

        self.training_configs = {"experiment": _SOURCE}
        self.task_type = []

    def load(self):
        from launcher.cloud.common import (
            WorkerSpec,
            DiscreteHP,
            ContinuousHP,
            HPO,
        )

        cloud_attributes = self.config.get("cloud_attributes", None)
        assert cloud_attributes, "`cloud_attributes` section is needed for remote mode."
        cfg_resources = cloud_attributes.get("resources", None)
        if cfg_resources:
            self.worker_spec = WorkerSpec(
                num_workers=cfg_resources.get("worker_num", _DEFAULT_WORKER_NUM),
                cpu=cfg_resources.get("worker_core", _DEFAULT_WORKER_CORE),
                memory=cfg_resources.get("worker_memory", _DEFAULT_WORKER_MEMORY),
                gpu=cfg_resources.get("worker_gpu_count", _DEFAULT_WORKER_GPU_COUNT),
            )

        cfg_hpo = cloud_attributes.get("hpo", None)
        if cfg_hpo:
            self.hpo = HPO(
                goal_metric=cfg_hpo.goal_metric,  # must parameter
                hp_list=[
                    DiscreteHP(**cfg_hp.params)
                    if cfg_hp.type == "DiscreteHP"
                    else ContinuousHP(**cfg_hp.params)
                    for cfg_hp in cfg_hpo.hyperparameters
                ],
                algorithm=cfg_hpo.get("algorithm", HPO.ALG_GRID),
            )
        self.datasets = cloud_attributes.get("standard_datasets", None)
        self.scene = cloud_attributes.get("scene", _DEFAULT_SCENE)
        self.user = str(cloud_attributes.user)
        self.server = cloud_attributes.get("server", None)
        self.packages = cloud_attributes.get("packages", None)
        self.code = cloud_attributes.get("code", None)

        self.training_configs["task_type"] = list(self.config.task_attributes.keys())[0]
        self.training_configs["task_type"] = self.config.tasks
        self.training_configs["datasets"] = self.config.datasets
        if self.config.datasets == "all":
            self.training_configs["datasets"] = ",".join(
                list(
                    self.config.task_attributes[self.config.tasks][
                        "dataset_attributes"
                    ].keys()
                )
            )

        self.training_configs["model_name"] = self.config.model

        run_type = self.config.get("run_type", "train+val")
        if run_type == "train+val":
            self.task_type = ["train"]

    def _get_session_maker(self):
        from launcher.cloud import (
            SessionMaker,
            # get_default_session_maker,
            get_pre_session_maker,
        )

        session_maker = get_pre_session_maker(user=self.user)
        if self.server:
            session_maker = SessionMaker(user=self.server, address=self.server)
        return session_maker

    def train(self):
        from launcher.cloud import TaskBuilder

        task_builder = TaskBuilder(
            source=_SOURCE,
            solution=_SOLUTION,
            scene_type=_SCENE_TYPE,
            scene=self.scene,
            biz_id="10",
            default_worker_spec=self.worker_spec,
        )

        module = task_builder.add_module(name=_MODULE_NAME)
        if self.code:
            module.set_code(self.code)

        model = module.add_model(name=_MODEL_NAME)
        if self.hpo:
            model.set_hpo(self.hpo)
        if self.packages:
            model.set_packages(self.packages)
        if self.training_configs:
            model.set_training_configs(self.training_configs)

        if self.datasets:
            task_builder.set_datasets(self.datasets)
        else:
            model.disable_sample()

        del self.config["cloud_attributes"]
        model["config_content"] = yaml.dump(json.loads(json.dumps(self.config)))

        session_maker = self._get_session_maker()
        result = session_maker.create(
            task_conf=task_builder.build(task_type=self.task_type), wait=True
        )
        print(result)
