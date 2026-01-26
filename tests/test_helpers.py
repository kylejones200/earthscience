"""
Helper functions for testing
"""

import pytest
import functools


def skip_if_not_implemented(module, attr_name):
    """
    Decorator to skip test if module doesn't have attribute.
    
    Usage:
        @skip_if_not_implemented(mymodule, 'function_name')
        def test_function():
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not hasattr(module, attr_name):
                pytest.skip(f"{module.__name__}.{attr_name} not implemented")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def requires_function(module, func_name):
    """
    Alternative decorator for skipping tests.
    
    Usage:
        @requires_function(mymodule, 'my_func')
        def test_my_func():
            ...
    """
    def decorator(test_func):
        if not hasattr(module, func_name):
            return pytest.mark.skip(
                reason=f"{module.__name__}.{func_name} not implemented"
            )(test_func)
        return test_func
    return decorator
