from typing import List, Optional

from textworld import EnvInfos
from textworld.gym.envs import TextworldBatchGymEnv, TextworldGymEnv


registry: dict[str, dict] = {}


def make(id: str):
    """Creates an environment previously registered with :meth:`textworld.gym.register_game`
    or a :meth:`textworld.gym.register_games`.

    To find all available environments use ``textworld.gym.registry.keys()`` for all valid ids.

    Args:
        id: A string for the environment id.

    Returns:
        An instance of the environment with wrappers applied.

    Raises:
        ValueError: If the ``id`` doesn't exist in the :attr:`registry`
    """
    if id not in registry:
        raise ValueError(f"Attempted to make an unknown environment '{id}'")

    klass = registry[id]["entry_point"]
    env = klass(**registry[id]["kwargs"])

    return env


def register_games(gamefiles: List[str],
                   request_infos: Optional[EnvInfos] = None,
                   batch_size: Optional[int] = None,
                   auto_reset: bool = False,
                   max_episode_steps: int = 50,
                   asynchronous: bool = True,
                   name: str = "",
                   **kwargs) -> str:
    """ Make an environment that will cycle through a list of games.

    Arguments:
        gamefiles:
            Paths for the TextWorld games (`*.ulx|*.z[1-8]`).
        request_infos:
            For customizing the information returned by this environment
            (see
            :py:class:`textworld.EnvInfos <textworld.core.EnvInfos>`
            for the list of available information).

            .. warning:: Only supported for TextWorld games (i.e., with a corresponding `*.json` file).
        batch_size:
            If provided, it indicates the number of games to play at the same time.
            By default, a single game is played at once.

            .. warning:: When `batch_size` is provided (even for batch_size=1), `env.step` expects
                         a list of commands as input and outputs a list of states. `env.reset` also
                         outputs a list of states.
        auto_reset:
            If `True`, each game *independently* resets once it is done (i.e., reset happens
            on the next `env.step` call).
            Otherwise, once a game is done, subsequent calls to `env.step` won't have any effects.
        max_episode_steps:
            Number of steps allocated to play each game. Once exhausted, the game is done.
        asynchronous:
            If `True`, games in the batch are played in parallel. Only when batch size is greater than one.
        name:
            Name for the new environment, i.e. "tw-{name}-v0". By default,
            the returned env_id is "tw-v0".

    Returns:
        The corresponding gym-compatible env_id to use.

    Example:

        >>> from textworld.generator import make_game, compile_game
        >>> options = textworld.GameOptions()
        >>> options.seeds = 1234
        >>> game = make_game(options)
        >>> game.extras["more"] = "This is extra information."
        >>> gamefile = compile_game(game)
        <BLANKLINE>
        >>> import gym
        >>> import textworld.gym
        >>> from textworld import EnvInfos
        >>> request_infos = EnvInfos(description=True, inventory=True, extras=["more"])
        >>> env_id = textworld.gym.register_games([gamefile], request_infos)
        >>> env = gym.make(env_id)
        >>> ob, infos = env.reset()
        >>> print(infos["extra.more"])
        This is extra information.

    """
    env_id = "tw-{}-v0".format(name) if name else "tw-v0"

    # If env already registered, bump the version number.
    if env_id in registry:
        base, _ = env_id.rsplit("-v", 1)
        versions = [int(env_id.rsplit("-v", 1)[-1]) for env_id in registry if env_id.startswith(base)]
        env_id = "{}-v{}".format(base, max(versions) + 1)

    entry_point = TextworldBatchGymEnv
    if batch_size is None:
        batch_size = 1
        entry_point = TextworldGymEnv

    registry[env_id] = dict(
        id=env_id,
        entry_point=entry_point,
        kwargs={
            'gamefiles': gamefiles,
            'request_infos': request_infos,
            'batch_size': batch_size,
            'asynchronous': asynchronous,
            'auto_reset': auto_reset,
            'max_episode_steps': max_episode_steps,
            **kwargs}
    )
    return env_id


def register_game(gamefile: str,
                  request_infos: Optional[EnvInfos] = None,
                  batch_size: Optional[int] = None,
                  auto_reset: bool = False,
                  max_episode_steps: int = 50,
                  asynchronous: bool = True,
                  name: str = "",
                  **kwargs) -> str:
    """ Make an environment for a particular game.

    Arguments:
        gamefile:
            Path for the TextWorld game (`*.ulx|*.z[1-8]`).
        request_infos:
            For customizing the information returned by this environment
            (see :py:class:`textworld.EnvInfos <textworld.core.EnvInfos>`
            for the list of available information).

            .. warning:: Only supported for TextWorld games (i.e., with a corresponding `*.json` file).
        batch_size:
            If provided, it indicates the number of games to play at the same time.
            By default, a single game is played at once.

            .. warning:: When `batch_size` is provided (even for batch_size=1), `env.step` expects
                         a list of commands as input and outputs a list of states. `env.reset` also
                         outputs a list of states.
        auto_reset:
            If `True`, each game *independently* resets once it is done (i.e., reset happens
            on the next `env.step` call).
            Otherwise, once a game is done, subsequent calls to `env.step` won't have any effects.
        max_episode_steps:
            Number of steps allocated to play each game. Once exhausted, the game is done.
        asynchronous:
            If `True`, games in the batch are played in parallel. Only when batch size is greater than one.
        action_space:
            The action space be used with OpenAI baselines.
            (see :py:class:`textworld.gym.spaces.Word <textworld.gym.spaces.text_spaces.Word>`).
        observation_space:
            The observation space be used with OpenAI baselines
            (see :py:class:`textworld.gym.spaces.Word <textworld.gym.spaces.text_spaces.Word>`).
        name:
            Name for the new environment, i.e. "tw-{name}-v0". By default,
            the returned env_id is "tw-v0".

    Returns:
        The corresponding gym-compatible env_id to use.

    Example:

        >>> from textworld.generator import make_game, compile_game
        >>> options = textworld.GameOptions()
        >>> options.seeds = 1234
        >>> game = make_game(options)
        >>> game.extras["more"] = "This is extra information."
        >>> gamefile = compile_game(game)
        <BLANKLINE>
        >>> import gym
        >>> import textworld.gym
        >>> from textworld import EnvInfos
        >>> request_infos = EnvInfos(description=True, inventory=True, extras=["more"])
        >>> env_id = textworld.gym.register_game(gamefile, request_infos)
        >>> env = gym.make(env_id)
        >>> ob, infos = env.reset()
        >>> print(infos["extra.more"])
        This is extra information.

    """
    return register_games(
        gamefiles=[gamefile],
        request_infos=request_infos,
        batch_size=batch_size,
        max_episode_steps=max_episode_steps,
        asynchronous=asynchronous,
        name=name,
        **kwargs
    )
