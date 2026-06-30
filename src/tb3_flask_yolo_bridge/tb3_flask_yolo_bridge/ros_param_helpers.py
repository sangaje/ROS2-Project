from rcl_interfaces.msg import ParameterDescriptor


TRUE_STRINGS = {'1', 'true', 't', 'yes', 'y', 'on'}
FALSE_STRINGS = {'0', 'false', 'f', 'no', 'n', 'off', ''}


def _warn_invalid(logger, name, value, default, expected_type):
    if logger is None:
        return
    logger.warn(
        f'invalid {expected_type} parameter {name}={value!r}; '
        f'using default {default!r}'
    )


def coerce_bool(value, default=False, *, name='parameter', logger=None):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in TRUE_STRINGS:
            return True
        if text in FALSE_STRINGS:
            return False
    _warn_invalid(logger, name, value, default, 'bool')
    return bool(default)


class FlexibleParameterNodeMixin:
    """Accept ROS launch/YAML parameter overrides even when their runtime type differs.

    launch_ros often sends LaunchConfiguration values as strings.  On ROS 2 Jazzy,
    declaring a parameter with a float/bool/int default can reject a string override
    before our code gets a chance to cast it.  Dynamic typing keeps declaration from
    crashing, while explicit bool parsing avoids Python's bool("false") == True trap.
    """

    def declare_parameter(self, name, value=None, descriptor=None, ignore_override=False):
        if descriptor is None:
            descriptor = ParameterDescriptor(dynamic_typing=True)
        return super().declare_parameter(name, value, descriptor, ignore_override)

    def declare_bool_parameter(self, name, default=False):
        value = self.declare_parameter(name, default).value
        return coerce_bool(
            value,
            default,
            name=name,
            logger=self.get_logger(),
        )
