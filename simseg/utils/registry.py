r""" Register certain module families. Refer to [mmcv](https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/registry.py).
"""

__all__ = ['Registry', 'build_from_cfg']


class Registry(object):
    """A registry to map strings to classes.

    Args:
        name (str): Registry name.
    """

    def __init__(self, name):
        self._name = name
        self._obj_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._obj_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def obj_dict(self):
        return self._obj_dict

    def get(self, key):
        return self._obj_dict.get(key, None)

    def has(self, key):
        if key in self._obj_dict.keys():
            return True
        return False

    def _register_obj(self, obj):
        """Register a object.

        Args:
            obj (:obj: callable): Callable object to be registered.
        """
        if not callable(obj):
            raise TypeError(f'object {str(obj)} must be callable')
        obj_name = obj.__name__
        if obj_name in self._obj_dict:
            raise KeyError(f'{obj_name} is already registered in {self.name}.')
        self._obj_dict[obj_name] = obj

    def register_obj(self, obj):
        self._register_obj(obj)
        return obj


def build_from_cfg(name, cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        name (str): Name of the object
        cfg (addict): Config dict of the object
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """

    assert isinstance(default_args, dict) or default_args is None

    obj = registry.get(name)
    if obj is None:
        raise KeyError(f'{name} is not in the {registry.name} registry. '
                       f'Choose among {list(registry.obj_dict.keys())}')

    if default_args is not None:
        for key, value in default_args.items():
            cfg.setdefault(key, value)

    return obj(cfg)
