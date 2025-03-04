if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import os
os.chdir('/home/src/your_codebase')
from tracking.mlflow import model_registry
from importlib import reload
import tracking.mlflow
reload(tracking.mlflow)
@custom
def transform_custom(*args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    exp_name = kwargs['exp_name']
    model_registry(exp_name)
    
    return "Registry Completed"


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
