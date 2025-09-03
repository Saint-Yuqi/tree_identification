from pytorch_lightning import callbacks
from omegaconf import DictConfig

def get_callbacks(callbacks_config: DictConfig):
    callbacks_list = []
    for callback in callbacks_config:
        callback_type = callback['type']
        params = callback.get('params', {})
        callback_class = getattr(callbacks, callback_type)
        callbacks_list.append(callback_class(**params))
    return callbacks_list
