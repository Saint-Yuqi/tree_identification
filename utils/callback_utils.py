from pytorch_lightning import callbacks
from omegaconf import DictConfig

#create a list of Lighning callbacks from config
def get_callbacks(callbacks_config: DictConfig):
    callbacks_list = []
    for callback in callbacks_config:
        callback_type = callback['type'] #name of callback class 
        params = callback.get('params', {}) 
        callback_class = getattr(callbacks, callback_type) #get callbacks
        callbacks_list.append(callback_class(**params))
    return callbacks_list
