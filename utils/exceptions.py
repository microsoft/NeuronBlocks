# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import sys


class BaseError(RuntimeError):
    """ Error base class

    """
    def __init__(self, arg, err_id=None):
        self.arg = arg
        self.err_id = err_id

    def __str__(self):
        if self.err_id is None:
            return self.arg
        else:
            return "error=%d, %s" % (self.err_id, self.arg)


class LayerConfigUndefinedError(BaseError):
    """ Errors occur when the corresponding configuration class of a layer is not defined

    """
    pass


class LayerUndefinedError(BaseError):
    """ Errors occur when some undefined layers are used

    """
    pass


class LayerDefineError(BaseError):
    """ (For developers) Errors occurs when there are some problems with the defined layers

    """
    pass


class ConfigurationError(BaseError):
    """ Errors occur when model configuration

    """
    pass


class InputError(BaseError):
    """ Error occur when the input to model is wrong

    """
    pass


class PreprocessError(BaseError):
    """ Error occur when the input to model is wrong

    """
    pass

