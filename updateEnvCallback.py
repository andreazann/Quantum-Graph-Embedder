from stable_baselines3.common.callbacks import BaseCallback


class UpdateEnvCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, env, eval_env, eval_freq, graph_training_set, graph_validation_set, verbose: int = 0):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.env = env
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.graph_training_set = graph_training_set
        self.graph_validation_set = graph_validation_set
        self.curr_train_graph = 0
        self.curr_eval_graph = 0
        self.call_counter = 1

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        
        self.curr_train_graph = (self.curr_train_graph+1) % len(self.graph_training_set)

        #print(f"### NOW TRAINING ON GRAPH {self.curr_train_graph+1} ###")

        self.env.update_source_graph(self.graph_training_set[self.curr_train_graph])

        if((self.call_counter % self.eval_freq) == 0):
            print(f"### NOW VALIDATING ON GRAPH {self.curr_eval_graph+1} ###")
            self.eval_env.update_source_graph(self.graph_validation_set[self.curr_eval_graph])
            self.curr_eval_graph = (self.curr_eval_graph+1) % len(self.graph_validation_set)

        self.call_counter = self.call_counter + 1 

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass