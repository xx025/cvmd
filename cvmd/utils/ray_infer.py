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
        runs_config: Dict[str, Any] = {},
        model_config: Dict[str, Any] = {},
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
        self.runs_config = runs_config
        self.model_config = model_config
        self.model_config["model"] = self.model
        self.handler = handler

    def infer(
        self,
        task: Any,
        *args,
        **kwds,
    ) -> Dict[str, Any]:
        return self.handler(task, self.model_config, self.runs_config, *args, **kwds)


def ray_infer_iter(
    InferActorCls,
    tasks: List[Any],
    *,
    num_actors: int = 6,
    num_cpus: int = 2,
    gpus_per_actor: Optional[float] = None,
    actor_kwargs: Optional[dict] = None,
    remote_method: str = "infer",
    remote_args: Optional[tuple] = None,
    remote_kwargs: Optional[dict] = None,
) -> Iterable[Any]:

    if not ray.is_initialized():
        ray.init()

    actor_kwargs = actor_kwargs or {}
    remote_args = remote_args or ()
    remote_kwargs = remote_kwargs or {}

    if gpus_per_actor is None:
        gpus_per_actor = 1.0 / num_actors

    RemoteInfer = ray.remote(num_gpus=gpus_per_actor, num_cpus=num_cpus)(InferActorCls)
    actors = [RemoteInfer.remote(**actor_kwargs) for _ in range(num_actors)]
    pool = ActorPool(actors)

    for r in pool.map_unordered(
        lambda a, t: getattr(a, remote_method).remote(t, *remote_args, **remote_kwargs),
        tasks,
    ):
        yield r
