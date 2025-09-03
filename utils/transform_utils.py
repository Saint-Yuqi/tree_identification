import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(transform_config, additional_targets={}):
    transforms_list = []
    for transform in transform_config:
        transform_type = transform['type']
        params = transform.get('params', {})
        try:
            transform_class = getattr(A, transform_type)
        except AttributeError:
            raise ValueError(f"Transform '{transform_type}' not found in Albumentations. "
                             f"Check if it was removed or renamed in v2.0+.")
        try:
            transforms_list.append(transform_class(**params))
        except Exception as e:
            raise ValueError(f"Failed to instantiate '{transform_type}' with params {params}: {e}")
    transforms_list.append(ToTensorV2())
    if not additional_targets:
        return A.Compose(transforms_list)
    return A.Compose(transforms_list, additional_targets=additional_targets)
