import os
import tempfile
from typing import Callable, Dict, Any, Iterable, List, Optional


from cvmd import build


try:
    # only import ray when needed
    import ray
    from ray.util.actor_pool import ActorPool
except ImportError:
    ray = None
    ActorPool = None


class InferActor:
    def __init__(
        self,
        runs_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        handler: Optional[Callable[..., Any]] = None,
    ):
        # print("Initializing InferActor...")
        # print(f"Model config: {model_config}")
        # print(f"Runs config: {runs_config}")
        self.model = build(
            model_config.get("model_name"),
            **model_config,
        )
        self.model.load_model()
        self.runs_config = runs_config or {}
        self.model_config = model_config or {}
        self.model_config["model"] = self.model
        self.handler = handler

    def infer(
        self,
        task: Any,
        *args,
        **kwds,
    ) -> Dict[str, Any]:
        return self.handler(task, self.model_config, self.runs_config, *args, **kwds)


def _ensure_ray_initialized(
    *,
    ray_address: Optional[str] = "local",
    ray_init_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    if ray is None:
        raise ImportError("ray is required for ray_infer_iter, please install ray first")

    if ray.is_initialized():
        return

    init_kwargs = {
        "include_dashboard": False,
        "ignore_reinit_error": True,
        "_temp_dir": os.path.join(tempfile.gettempdir(), "cvmd-ray", str(os.getpid())),
    }
    init_kwargs.update(ray_init_kwargs or {})

    if ray_address is not None and "address" not in init_kwargs:
        init_kwargs["address"] = ray_address

    ray.init(**init_kwargs)


def ray_infer_iter(
    InferActorCls,
    tasks: List[Any],
    *,
    num_actors: Optional[int] = None,
    num_cpus: float = 2.0,
    gpus_per_actor: float = 0.25,
    actor_kwargs: Optional[dict] = None,
    remote_method: str = "infer",
    remote_args: Optional[tuple] = None,
    remote_kwargs: Optional[dict] = None,
    ray_address: Optional[str] = "local",
    ray_init_kwargs: Optional[Dict[str, Any]] = None,
) -> Iterable[Any]:
    """Run inference tasks through a Ray actor pool.

    By default, each Python process starts an isolated local Ray runtime to avoid
    collisions between concurrent local jobs. Pass ray_address="auto" or an
    explicit address in ray_init_kwargs to connect to an existing Ray cluster.
    """

    _ensure_ray_initialized(
        ray_address=ray_address,
        ray_init_kwargs=ray_init_kwargs,
    )

    actor_kwargs = actor_kwargs or {}
    remote_args = remote_args or ()
    remote_kwargs = remote_kwargs or {}

    total_gpus = ray.cluster_resources().get("GPU", 0)
    if total_gpus > 0:
        if num_actors is None:
            # Standard Ray pattern: Fix resource requirement per actor, 
            # and dynamically calculate the number of actors to fill the cluster.
            num_actors = int(total_gpus / gpus_per_actor)
    else:
        gpus_per_actor = 0
        if num_actors is None:
            num_actors = 1

    num_actors = max(1, num_actors)

    RemoteInfer = ray.remote(num_gpus=gpus_per_actor, num_cpus=num_cpus)(InferActorCls)
    actors = [RemoteInfer.remote(**actor_kwargs) for _ in range(num_actors)]
    pool = ActorPool(actors)

    for r in pool.map_unordered(
        lambda a, t: getattr(a, remote_method).remote(t, *remote_args, **remote_kwargs),
        tasks,
    ):
        yield r
