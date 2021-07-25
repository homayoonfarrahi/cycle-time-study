from gym.envs.registration import register

register(
    id='SelfAwareReacherBulletEnv-v0',
    entry_point='pybullet_env_mods.envs:SelfAwareReacherBulletEnv',
    max_episode_steps=150,
    reward_threshold=18.0,
)

register(
    id='HighFreqReacherBulletEnv-v0',
    entry_point='pybullet_env_mods.envs:HighFreqReacherBulletEnv',
    max_episode_steps=1200,
    reward_threshold=18.0,
)

register(
    id='HighFreqInvertedDoublePendulumBulletEnv-v0',
    entry_point='pybullet_env_mods.envs:HighFreqInvertedDoublePendulumBulletEnv',
    max_episode_steps=4000,
    reward_threshold=9100.0,
)
