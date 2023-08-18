from metaflow import FlowSpec, step, batch
from decorators import gpu_profile, pip

DATA_URL = "s3://outerbounds-datasets/ubiquant/investment_ids"
RESOURCES = dict(memory=16000, cpu=4, use_tmpfs=True, tmpfs_size=4000)

class RayXGBoostGPU(FlowSpec):

    n_files = 500
    n_cpu = RESOURCES['cpu']
    s3_url = DATA_URL

    @step
    def start(self):
        self.next(self.train)

    @gpu_profile(interval=1)
    @batch(**RESOURCES, image='eddieob/ray-demo:xgboost-gpu')
    @step
    def train(self):
        
        import os
        import ray
        from metaflow import S3
        from table_loader import load_table
        from xgb_example import load_data, fit_model
        
        # Initialize ray driver on the cluster @ray_parallel created.
        ray.init()

        # Load many files from S3 using Metaflow + PyArrow. 
        # Then convert to Ray.dataset.
        table = load_table(self.s3_url, self.n_files, drop_cols=['row_id'])
        train_dataset, valid_dataset = load_data(table=table)

        # Store checkpoints in S3, versioned by Metaflow run_id.
        self.checkpoint_path = os.path.join(DATATOOLS_S3ROOT, current.flow_name, current.run_id, 'ray_checkpoints')
        self.result = fit_model(
            train_dataset, valid_dataset, 
            n_cpu = self.n_cpu,
            n_gpu = self.n_gpu,
            objective='reg:squarederror',
            eval_metric=['rmse'],
            run_config_storage_path = self.checkpoint_path,
        )

        self.next(self.end)

    @step
    def end(self):
        print(self.result)

if __name__ == '__main__':
    RayXGBoostGPU()