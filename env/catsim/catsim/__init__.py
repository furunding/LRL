from gym.envs.registration import register

register(
    id='catsim-v0',
    entry_point='catsim.envs:CatSimSA',
)