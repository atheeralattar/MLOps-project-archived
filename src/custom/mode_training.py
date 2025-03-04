if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import os
os.chdir('/home/src/your_codebase')
from modeling.modeling import train
from importlib import reload
import modeling.modeling
reload(modeling.modeling)
#from tracking.mlflow import mlflow_default_logging

@custom
def transform_custom(data, *args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    tracking_uri = kwargs['tracking_uri']
    train(data[:1000],tracking_uri)
    return 'training complete'


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
