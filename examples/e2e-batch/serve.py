from ray.train.xgboost import XGBoostPredictor
from ray import serve
from ray.serve import PredictorDeployment
from ray.serve.http_adapters import pandas_read_json

from metaflow import Flow

def select_from_checkpoint_registry(flow_name = "Train"):
    flow = Flow(flow_name)
    run = flow.latest_successful_run
    result = run.data.result
    return result.checkpoint

checkpoint = select_from_checkpoint_registry()

serve.run(
    PredictorDeployment.options(name="XGBoostService").bind(
        XGBoostPredictor, checkpoint, http_adapter=pandas_read_json
    )
)