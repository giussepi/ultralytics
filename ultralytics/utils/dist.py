# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import shutil
import socket
import sys
import tempfile

from . import USER_CONFIG_DIR
from .torch_utils import TORCH_1_9


def find_free_network_port() -> int:
    """
    Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]  # port


def generate_ddp_file(trainer, /, *, model_yaml_file='', weights_path=''):
    """Generates a DDP file and returns its file name."""
    assert isinstance(model_yaml_file, str), type(model_yaml_file)
    assert isinstance(weights_path, str), type(weights_path)

    module, name = f'{trainer.__class__.__module__}.{trainer.__class__.__name__}'.rsplit('.', 1)

    if not os.path.isfile(model_yaml_file):
        content = f"""
# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)
overrides = {vars(trainer.args)}

if __name__ == "__main__":
    from {module} import {name}
    from ultralytics.utils import DEFAULT_CFG_DICT

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # handle the extra key 'save_dir'
    trainer = {name}(cfg=cfg, overrides=overrides)
    results = trainer.train()
"""
    else:
        content = f"""
# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)
overrides = {vars(trainer.args)}

if __name__ == "__main__":
    import os

    from {module} import {name}
    from ultralytics import YOLO
    from ultralytics.utils import DEFAULT_CFG_DICT

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # handle the extra key 'save_dir'
    trainer = {name}(cfg=cfg, overrides=overrides)

    model = YOLO('{model_yaml_file}')

    if os.path.isfile('{weights_path}'):
        model = model.load(('{weights_path}'))

    # Tweak to make it work when it's not resuming and a custom model has been provided
    # inspired on ultralytics/engine/model.py -> Model.train
    if not overrides.get('resume'):  # manually set model only if not resuming
        # note: passing the detection model and its config (custom_yolov8n.yaml) as dict
        trainer.model = trainer.get_model(weights=model.model if model.ckpt else None, cfg=model.model.yaml)
        model.model = trainer.model  # maybe this line is not needed

    results = trainer.train()
"""

    (USER_CONFIG_DIR / 'DDP').mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='_temp_',
                                     suffix=f'{id(trainer)}.py',
                                     mode='w+',
                                     encoding='utf-8',
                                     dir=USER_CONFIG_DIR / 'DDP',
                                     delete=False) as file:
        file.write(content)

    return file.name


def generate_ddp_command(
        world_size, trainer, /, *, model_yaml_file='', weights_path=''):
    """Generates and returns command for distributed training."""
    assert isinstance(model_yaml_file, str), type(model_yaml_file)
    assert isinstance(weights_path, str), type(weights_path)

    import __main__  # noqa local import to avoid https://github.com/Lightning-AI/lightning/issues/15218
    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # remove the save_dir
    file = generate_ddp_file(
        trainer, model_yaml_file=model_yaml_file, weights_path=weights_path)
    # file = generate_ddp_file(trainer)
    dist_cmd = 'torch.distributed.run' if TORCH_1_9 else 'torch.distributed.launch'
    port = find_free_network_port()
    cmd = [sys.executable, '-m', dist_cmd, '--nproc_per_node', f'{world_size}', '--master_port', f'{port}', file]
    return cmd, file


def ddp_cleanup(trainer, file):
    """Delete temp file if created."""
    if f'{id(trainer)}.py' in file:  # if temp_file suffix in file
        os.remove(file)
