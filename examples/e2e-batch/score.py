from metaflow import FlowSpec, step, trigger_on_finish, current, Parameter
from base import TabularBatchPrediction

PARENT_FLOW_1 = "Train"
PARENT_FLOW_2 = "Tune"
TRIGGERS = [PARENT_FLOW_1, PARENT_FLOW_2]

@trigger_on_finish(flows=TRIGGERS)
class Score(FlowSpec, TabularBatchPrediction):

    upstream_flow = Parameter("upstream", help="Upstream flow name", default=TRIGGERS[0], type=str)

    def _fetch_eval_set(self):
        _, valid_dataset, test_dataset = self.load_dataset()
        true_targets = valid_dataset.select_columns(cols=["target"]).to_pandas()
        return true_targets, test_dataset

    @step
    def start(self):
        import pandas as pd

        try:
            upstream_flow = current.trigger.run.parent.pathspec
            assert upstream_flow in TRIGGERS
        except AssertionError:
            print(
                "Score flow can only be triggered by Train or Tune flow. Please check your flow of flows."
            )
            exit(1)
        except AttributeError:
            upstream_flow = self.upstream_flow
            print(
                f"Current run was not triggered. Defaulting to {upstream_flow}."
            )

        self.setup()
        true_targets, test_dataset = self._fetch_eval_set()
        preds = self.batch_predict(
            dataset=test_dataset,
            checkpoint=self.load_checkpoint(flow_name=upstream_flow)
        ).to_pandas()
        self.score_results = pd.concat([true_targets, preds], axis=1)
        self.next(self.end)

    @step
    def end(self):
        for threshold in [0.25, 0.5, 0.75]:
            self.score_results[f"pred @ {threshold}"] = self.score_results[
                "predictions"
            ].apply(lambda x: 1 if x > threshold else 0)
            print(
                "Accuracy with threshold @ {threshold}: {val}%".format(
                    threshold=threshold,
                    val=round(
                        100
                        * (
                            self.score_results["target"]
                            == self.score_results[f"pred @ {threshold}"]
                        ).sum()
                        / len(self.score_results),
                        2,
                    ),
                )
            )
        print(f"""

            Access result:

            from metaflow import Run
            run = Run('{current.flow_name}/{current.run_id}')
            df = run.data.score_results
        """)


if __name__ == "__main__":
    Score()
