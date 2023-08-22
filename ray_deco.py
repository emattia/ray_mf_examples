import os
import sys
import time
import json
import subprocess
from pathlib import Path
from metaflow.unbounded_foreach import UBF_CONTROL
from metaflow.plugins.parallel_decorator import ParallelDecorator, _local_multinode_control_task_step_func


RAY_JOB_COMPLETE_VAR = 'ray_job_completed'
RAY_NODE_STARTED_VAR = 'node_started'
CONTROL_TASK_S3_ID = 'control'


class RayParallelDecorator(ParallelDecorator):


    name = "ray_parallel"
    defaults = {"main_port": None, "worker_polling_freq": 10}
    IS_PARALLEL = True


    def task_decorate(
        self, step_func, flow, graph, retry_count, max_user_code_retries, ubf_context
    ):

        from functools import partial
        from metaflow import S3

        def _empty_worker_task():
            pass # local case

        def _worker_heartbeat(polling_freq=self.attributes["worker_polling_freq"], var=RAY_JOB_COMPLETE_VAR):
            s3 = S3(run=flow)
            while not json.loads(s3.get(CONTROL_TASK_S3_ID).blob)[var]:
                time.sleep(polling_freq)

        def _control_task_wrapper(step_func, flow, var=RAY_JOB_COMPLETE_VAR):
            s3 = S3(run=flow)
            step_func()
            s3.put(CONTROL_TASK_S3_ID, json.dumps({var: True}))

        if os.environ.get("METAFLOW_RUNTIME_ENVIRONMENT", "local") == "local":
            if ubf_context == UBF_CONTROL:
                env_to_use = getattr(self.environment, "base_env", self.environment)
                return partial(
                    _local_multinode_control_task_step_func,
                    flow,
                    env_to_use, 
                    step_func,
                    retry_count,
                )    
            return partial(_empty_worker_task)
        else:
            self.setup_distributed_env(flow, ubf_context)
            if ubf_context == UBF_CONTROL:
                return partial(_control_task_wrapper, step_func=step_func, flow=flow)
            return partial(_worker_heartbeat)


    def setup_distributed_env(self, flow, ubf_context):
        self.ensure_ray_air_installed()
        py_cli_path = Path(sys.executable)
        if py_cli_path.is_symlink():
            py_cli_path = os.readlink(py_cli_path)
        ray_cli_path = os.path.join(py_cli_path.split('bin')[0], 'bin', 'ray')
        setup_ray_distributed(self.attributes["main_port"], ray_cli_path, flow, ubf_context)


    def ensure_ray_air_installed(self):
        try:
            import ray
        except ImportError:
            print("Ray is not installed. Installing latest version of ray-air package.")
            subprocess.run([sys.executable, "-m", "pip", "install", "-U", "ray[air]"])


def setup_ray_distributed(
    main_port=None, 
    ray_cli_path=None, 
    run=None, 
    ubf_context=None
):

    import ray
    import json
    import socket
    from metaflow import S3, current

    # Why are deco.task_pre_step and deco.task_decorate calls in the same loop?
    # https://github.com/Netflix/metaflow/blob/76eee802cba1983dffe7e7731dd8e31e2992e59b/metaflow/task.py#L553
        # The way this runs now causes these current.parallel variables to be defaults on all nodes,
        # since AWS Batch decorator task_pre_step hasn't run prior to the above task_decorate call.
    # num_nodes = current.parallel.num_nodes
    # node_index = current.parallel.node_index

    # AWS Batch-specific workaround.
    num_nodes = int(os.environ["AWS_BATCH_JOB_NUM_NODES"])
    node_index = os.environ["AWS_BATCH_JOB_NODE_INDEX"]
    node_key = os.path.join(RAY_NODE_STARTED_VAR, "node_%s.json" % node_index)

    # Similar to above comment, 
    # better to use current.parallel.main_ip instead of this conditional block, 
    # but this seems to require a change to the main loop in metaflow.task. 
    if ubf_context == UBF_CONTROL:
        local_ips = socket.gethostbyname_ex(socket.gethostname())[-1]
        main_ip = local_ips[0]
    else: 
        main_ip = os.environ['AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS']
    
    try:
        main_port = main_port or (6379 + abs(int(current.run_id)) % 1000)
    except:
        main_port = 6379

    s3 = S3(run=run)

    if ubf_context == UBF_CONTROL:
        runtime_start_result = subprocess.run(
            [
                ray_cli_path,
                "start",
                "--head",
                "--node-ip-address",
                main_ip,
                "--port",
                str(main_port),
            ]
        )
        s3.put('control', json.dumps({RAY_JOB_COMPLETE_VAR: False}))
    else:
        node_ip_address = ray._private.services.get_node_ip_address()
        runtime_start_result = subprocess.run(
            [
                ray_cli_path,
                "start",
                "--node-ip-address",
                node_ip_address,
                "--address",
                "%s:%s" % (main_ip, main_port),
            ]
        )
    if runtime_start_result.returncode != 0:
        raise Exception("Ray runtime failed to start on node %s" % node_index)
    else:
        s3.put(node_key, json.dumps({'node_started': True}))

    def _num_nodes_started(path=RAY_NODE_STARTED_VAR):
        objs = s3.get_recursive([path])
        num_started = 0
        for obj in objs:
            obj = json.loads(obj.text)
            if obj['node_started']:
                num_started += 1
            else:
                raise Exception("Node {} failed to start Ray runtime".format(node_index))
        return num_started
    
    # poll until all workers have joined the cluster
    if ubf_context == UBF_CONTROL:
        while _num_nodes_started() < num_nodes:
            time.sleep(10)

    s3.close()