# scripts/wrappers/fixed_challenge_wrapper.py
from gym import Env
from CybORG.Agents.Wrappers import BaseWrapper


class FixedChallengeWrapper(Env, BaseWrapper):
    def __init__(self, agent_name: str, env, reward_threshold=None, max_steps=None):
        super().__init__(env)
        self.agent_name = agent_name
        self.env = env  # 直接使用基础环境

        self.reward_threshold = reward_threshold
        self.max_steps = max_steps
        self.step_counter = 0

    def step(self, action=None):
        # 使用基础环境的 step 方法
        result = self.env.step(agent=self.agent_name, action=action)

        self.step_counter += 1

        # 转换为 gym 格式
        obs = result.observation
        reward = result.reward
        done = result.done
        info = {
            'action_space': getattr(result, 'action_space', {}),
            'original_reward': reward
        }

        if self.max_steps is not None and self.step_counter >= self.max_steps:
            done = True

        return obs, reward, done, info

    def reset(self, seed=None):
        """修复 reset 方法"""
        self.step_counter = 0

        # 重置环境
        result = self.env.reset(agent=self.agent_name)

        # 返回 observation
        return result.observation

    # 保持其他方法不变
    def get_attr(self, attribute: str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        return self.env.get_observation(agent)

    def get_agent_state(self, agent: str):
        return self.env.get_agent_state(agent)

    def get_action_space(self, agent=None):
        return self.env.get_action_space(self.agent_name)

    def get_last_action(self, agent):
        return self.get_attr('get_last_action')(agent)

    def get_ip_map(self):
        return self.get_attr('get_ip_map')()

    def get_rewards(self):
        return self.get_attr('get_rewards')()