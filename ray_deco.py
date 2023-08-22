import os
import sys
import time
import json
import socket
import subprocess
from pathlib import Path
from functools import partial
import ray
from metaflow.unbounded_foreach import UBF_CONTROL
from metaflow.plugins.parallel_decorator import (
    ParallelDecorator,
    _local_multinode_control_task_step_func,
)

RAY_JOB_COMPLETE_VAR = "ray_job_completed"
RAY_NODE_STARTED_VAR = "node_started"
CONTROL_TASK_STATUS_KEY = "control"


class RayParallelDecorator(ParallelDecorator):

    name = "ray_parallel"
    defaults = {"main_port": None, "worker_polling_freq": 10}
    IS_PARALLEL = True

    def task_decorate(
        self, step_func, flow, graph, retry_count, max_user_code_retries, ubf_context
    ):

        from metaflow import S3

        def _empty_worker_task():
            pass  # local case

        def _worker_heartbeat(
            polling_freq=self.attributes["worker_polling_freq"],
            var=RAY_JOB_COMPLETE_VAR,
        ):
            s3 = S3(run=flow)
            while not json.loads(s3.get(CONTROL_TASK_STATUS_KEY).blob)[var]:
                time.sleep(polling_freq)

        def _control_task_wrapper(step_func, flow, var=RAY_JOB_COMPLETE_VAR):
            s3 = S3(run=flow)
            # sidecar subprocess that polls for active workers
            watcher = NodeParticipationWatcher(
                s3, current.num_nodes, "enforce_participation"
            )
            # and throws fatal error when more than n% of workers have died
            step_func()
            watcher.end()
            s3.put(CONTROL_TASK_STATUS_KEY, json.dumps({var: True}))

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
        ray_cli_path = os.path.join(py_cli_path.split("bin")[0], "bin", "ray")
        setup_ray_distributed(
            self.attributes["main_port"], ray_cli_path, flow, ubf_context
        )

    def ensure_ray_air_installed(self):
        try:
            import ray
        except ImportError:
            print("Ray is not installed. Installing latest version of ray-air package.")
            subprocess.run([sys.executable, "-m", "pip", "install", "-U", "ray[air]"])


def setup_ray_distributed(
    main_port=None, ray_cli_path=None, run=None, ubf_context=None
):

    import ray
    from metaflow import S3, current

    num_nodes = int(os.environ["AWS_BATCH_JOB_NUM_NODES"])
    node_index = os.environ["AWS_BATCH_JOB_NODE_INDEX"]
    node_key = os.path.join(RAY_NODE_STARTED_VAR, "node_%s.json" % node_index)
    current._update_env({"num_nodes": num_nodes, "node_index": node_index})
    if ubf_context == UBF_CONTROL:
        local_ips = socket.gethostbyname_ex(socket.gethostname())[-1]
        main_ip = local_ips[0]
    else:
        main_ip = os.environ["AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS"]
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
        s3.put("control", json.dumps({RAY_JOB_COMPLETE_VAR: False}))
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
        s3.put(node_key, json.dumps({"node_started": True}))

    # def _num_nodes_started(path=RAY_NODE_STARTED_VAR):
    #     objs = s3.get_recursive([path])
    #     num_started = 0
    #     for obj in objs:
    #         obj = json.loads(obj.text)
    #         if obj['node_started']:
    #             num_started += 1
    #         else:
    #             raise Exception("Node {} failed to start Ray runtime".format(node_index))
    #     return num_started

    # poll until all workers have joined the cluster
    if ubf_context == UBF_CONTROL:
        NodeParticipationWatcher(s3, current.num_nodes, "wait_until_all_started")
        # while _num_nodes_started(s3=s3, node_index=node_index) < num_nodes:
        #     time.sleep(10)

    s3.close()


def _num_nodes_started(path=RAY_NODE_STARTED_VAR, s3=None, node_index=None):
    objs = s3.get_recursive([path])
    num_started = 0
    for obj in objs:
        obj = json.loads(obj.text)
        if obj["node_started"]:
            num_started += 1
        elif node_index is None:
            my_ip = socket.gethostbyname_ex(socket.gethostname())[-1][0]
            raise Exception("Node {} failed to start Ray runtime".format(my_ip))
        else:
            raise Exception("Node {} failed to start Ray runtime".format(node_index))
    return num_started


from threading import Thread


class NodeParticipationWatcher(object):
    def __init__(self, s3_client, expected_num_nodes, target, polling_freq=10):
        self.s3 = s3_clients
        self.expected_num_nodes = expected_num_nodes
        self.polling_freq = polling_freq
        if target == "wait_until_all_started":
            self._thread = Thread(target=self._wait_until_all_started)
        elif target == "enforce_participation":
            self._thread = Thread(target=self._enforce_participation)
        self.is_alive = True
        self._thread.start()

    def end(self):
        self.is_alive = False

    def _wait_until_all_started(self):
        while self._num_nodes_started() < self.expected_num_nodes:
            time.sleep(self.polling_freq)
        self.is_alive = False

    def _enforce_participation(self):
        while self.is_alive:
            if self._num_nodes_started() < self.expected_num_nodes:
                self.is_alive = False
                self._kill_run(self._num_nodes_started())
            time.sleep(self.polling_freq)

    def _num_nodes_started(self, path=RAY_NODE_STARTED_VAR):
        return len(ray.nodes())

    def _kill_run(self, n):
        raise MetaflowException(
            "Exiting the run because not all nodes are active, only see {} out of {} expected.".format(
                n, self.expected_num_nodes
            )
        )
