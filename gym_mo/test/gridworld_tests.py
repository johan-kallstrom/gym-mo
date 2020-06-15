import unittest

from gym_mo.envs.gridworlds import gridworld_base
from gym_mo.envs.gridworlds.gridworld_base import GridObject, GridAgent, PixelRLAgent, ScrollerAgent, Viewport, Gridworld


class GridObjectTestCase(unittest.TestCase):
    """Test cases for functions of the GridObject class."""

    def setUp(self):
        self.grid_object = GridObject(is_walkable=True, is_consumed=True, reward_on_encounter=10, color=(0.0, 0.0, 255.0), idx=1)

    def test_init(self):
        """Intention: Test the init function of the class.

        Oracle: Asserts that the class attributes are according to the input in setUp.
        """
        self.assertEqual(self.grid_object.is_walkable, True, 'GridObject init is_walkable failed')
        self.assertEqual(self.grid_object.is_consumed, True, 'GridObject init is_consumed failed')
        self.assertEqual(self.grid_object.reward_on_encounter, 10, 'GridObject init reward_on_encounter failed')
        self.assertEqual(self.grid_object.color, (0.0, 0.0, 255.0), 'GridObject init color failed')
        self.assertEqual(self.grid_object.idx, 1, 'GridObject init idx failed')

class GridAgentTestCase(unittest.TestCase):
    """Test cases for functions of the GridAgent class."""

    def setUp(self):
        self.agent = GridAgent(is_walkable=False, is_consumed=False, reward_on_encounter=0, color=(0.0, 0.0, 255.0))

    def test_init(self):
        """Intention: Test the init function of the class.

            Oracle: Asserts that the class attributes are according to the input in setUp and expected initializations in init.
        """
        self.assertEqual(self.agent.is_walkable, False, 'GridAgent init is_walkable failed')
        self.assertEqual(self.agent.is_consumed, False, 'GridAgent init is_consumed failed')
        self.assertEqual(self.agent.reward_on_encounter, 0, 'GridAgent init reward_on_encounter failed')
        self.assertEqual(self.agent.color, (0.0, 0.0, 255.0), 'GridAgent init color failed')
        self.assertEqual(self.agent.idx, None, 'GridAgent init idx failed')
        self.assertEqual(self.agent.position, None, 'GridAgent init position failed')
        self.assertEqual(self.agent.reward, 0, 'GridAgent init reward failed')
        self.assertEqual(self.agent.observation_color, (0.0, 0.0, 255.0), 'GridAgent init observation_color failed')

    def test_reset_reward(self):
        """Intention: Test the function for resetting the agent's reward to 0.

        Oracle: Asserts that the agent's reward changes from 100 to 0 after calling reset_reward.
        """
        self.agent.reward = 100
        self.agent.reset_reward()
        self.assertEqual(self.agent.reward, 0, 'GridAgent reset_reward failed')

    def test_set_position(self):
        """Intention: Test the function for setting the agent's position.

        Oracle: Asserts that the agent's position is according to the input argument after calling set_position.
        """
        self.agent.set_position([2,3])
        self.assertEqual(self.agent.position, [2,3], 'GridAgent set_position failed')

class PixelRLAgentTestCase(unittest.TestCase):
    """Test cases for functions of the PixelRLAgent class."""

    def setUp(self):
        class Actor:
            def __init__(self, action):
                self.action = action
            def __call__(self, obs):
                return [self.action]

        self.env = Gridworld(map=gridworld_base.TEST_MAP, object_mapping=gridworld_base.TEST_MAPPING)
        self.actor = Actor(1)
        self.agent = PixelRLAgent(act=self.actor, is_walkable=False, is_consumed=False, reward_on_encounter=0, color=(0.0, 0.0, 255.0), idx=1)

    def test_init(self):
        """Intention: Test the init function of the class.

        Oracle: Asserts that the class attributes are according to the input in setUp and expected initializations in init.
        """
        self.assertEqual(self.agent.act, self.actor, 'PixelRLAgent init act failed')
        self.assertEqual(self.agent.is_walkable, False, 'PixelRLAgent init is_walkable failed')
        self.assertEqual(self.agent.is_consumed, False, 'PixelRLAgent init is_consumed failed')
        self.assertEqual(self.agent.reward_on_encounter, 0, 'PixelRLAgent init reward_on_encounter failed')
        self.assertEqual(self.agent.color, (0.0, 0.0, 255.0), 'PixelRLAgent init color failed')
        self.assertEqual(self.agent.idx, 1, 'PixelRLAgent init idx failed')
        self.assertEqual(self.agent.position, None, 'PixelRLAgent init position failed')
        self.assertEqual(self.agent.reward, 0, 'PixelRLAgent init reward failed')
        self.assertEqual(self.agent.observation_color, (0.0, 0.0, 255.0), 'PixelRLAgent init observation_color failed')

    def test_step_left(self):
        """Intention: Test the step function of the class, check that agent moves 1 step to the left when action is 1 (set in Actor in setUp).

        Oracle: Asserts that the first element of agent.position is decremented by 1 after calling step.
        """
        self.agent.set_position([4,1])
        self.agent.step(self.env)
        self.assertEqual(self.agent.position, [3,1], 'PixelRLAgent step failed')

class ScrollerAgentTestCase(unittest.TestCase):
    """Test cases for functions of the ScrollerAgent class."""

    def setUp(self):
        self.env = Gridworld(map=gridworld_base.TEST_MAP, object_mapping=gridworld_base.TEST_MAPPING)
        self.agent = ScrollerAgent(random_walk=False, 
                                   direction=ScrollerAgent.GOING_UP,
                                   step_delay=0, 
                                   step_delay_init=[0,0],
                                   random_init=None,
                                   is_walkable=True, 
                                   is_consumed=False, 
                                   reward_on_encounter=0, 
                                   color=(0.0, 255.0, 0.0), 
                                   idx=2)

    def test_init(self):
        """Intention: Test the init function of the class.

        Oracle: Asserts that the class attributes are according to the input in setUp and expected initializations in init.
        """
        self.assertEqual(self.agent.random_walk, False, 'ScrollerAgent init random_walk failed')
        self.assertEqual(self.agent.direction[0], 0, 'ScrollerAgent init direction failed')
        self.assertEqual(self.agent.direction[1], 1, 'ScrollerAgent init direction failed')
        self.assertEqual(self.agent.step_delay, False, 'ScrollerAgent init step_delay failed')
        self.assertEqual(self.agent.step_delay_init[0], 0, 'ScrollerAgent init step_delay_init failed')
        self.assertEqual(self.agent.step_delay_init[1], 0, 'ScrollerAgent init step_delay_init failed')
        self.assertEqual(self.agent.random_init, None, 'ScrollerAgent init random_init failed')
        self.assertEqual(self.agent.is_walkable, True, 'ScrollerAgent init is_walkable failed')
        self.assertEqual(self.agent.is_consumed, False, 'ScrollerAgent init is_consumed failed')
        self.assertEqual(self.agent.reward_on_encounter, 0, 'ScrollerAgent init reward_on_encounter failed')
        self.assertEqual(self.agent.color, (0.0, 255.0, 0.0), 'ScrollerAgent init color failed')
        self.assertEqual(self.agent.idx, 2, 'ScrollerAgent init idx failed')
        self.assertEqual(self.agent.position, None, 'ScrollerAgent init position failed')
        self.assertEqual(self.agent.reward, 0, 'ScrollerAgent init reward failed')
        self.assertEqual(self.agent.observation_color, (0.0, 0.0, 255.0), 'ScrollerAgent init observation_color failed')

    def test_set_new_direction(self):
        """Intention: Test that the function for changing the agent's direction of movement works.

        Oracle: Asserts that the agent's direction changes from UP (set in setUp) to DOWN (-1 in second element) when set_new_direction is called.
        """
        self.agent.set_position([7,1])
        self.agent.set_new_direction()
        self.assertEqual(self.agent.direction[1], -1, 'ScrollerAgent set new direction failed, did not switch direction')

    def test_step(self):
        """Intention: Test the step function of the class, check that agent changes direction when hitting a wall (based on MAP set in setUp).

        Oracle: Asserts that the agent's direction changes from UP (set in setUp) to DOWN (-1 in second element) when step is called.
        """       
        self.agent.set_position([7,1])
        self.agent.step(self.env)
        self.assertEqual(self.agent.direction[1], -1, 'ScrollerAgent step failed, did not bounce against wall')

    def test_reset(self):
        """Intention: Test the reset function of the class, check that step_delay is reset.

        Oracle: Asserts that the step_delay is reset by first setting it to invalid value -1 and then calling reset to seit change.
        """   
        self.agent.step_delay = -1
        self.agent.reset()
        self.assertNotEqual(self.agent.step_delay, -1, 'ScrollerAgent reset failed, did not reset step_delay')

class ViewportTestCase(unittest.TestCase):
    """Test cases for functions of the Viewport class."""

    def test_init(self):
        """Intention: Test the init function of the class.

        Oracle: Asserts that the class attributes are according to the input in setUp.
        """
        viewport = Viewport(viewport_width=800, viewport_height=800, view_port_object_delta=0.2)
        self.assertEqual(viewport.width, 800, 'Viewport init viewport_width failed')
        self.assertEqual(viewport.height, 800, 'Viewport init viewport_height failed')
        self.assertEqual(viewport.object_delta, 0.2, 'Viewport init view_port_object_delta failed')

    def test_default_settings(self):
        """Intention: Test the default settings of the class.

        Oracle: Asserts that the class attributes are according specification when creating class instance without input values.
        """
        viewport = Viewport()
        self.assertEqual(viewport.width, 600, 'Viewport init default viewport_width failed')
        self.assertEqual(viewport.height, 600, 'Viewport init default viewport_height failed')
        self.assertEqual(viewport.object_delta, 0.1, 'Viewport init default view_port_object_delta failed')

class GridworldTestCase(unittest.TestCase):
    """Test cases for functions of the Gridworld class."""

    def setUp(self):
        self.env = Gridworld(map=gridworld_base.TEST_MAP, object_mapping=gridworld_base.TEST_MAPPING)

    def test_get_new_pos_from_action(self):
        """Intention: Test the function for getting a new position based on action taken.

        Oracle: Asserts that the first element is unchanged and the second incremented by 1 when action is 4 (move up).
        """
        old_pos = [2,5]
        new_pos = self.env.get_new_pos_from_action(action=4, old_pos=old_pos)
        self.assertEqual(new_pos[0], 2, 'Gridworld get_new_pos_from_action failed')
        self.assertEqual(new_pos[1], 6, 'Gridworld get_new_pos_from_action failed')

    def test_step(self):
        """Intention: Test the step function of the class, check that received reward is -6 when walking into a wall (according to MAP in setUp).

        Oracle: Asserts that the reward is -6 when we move up (action 4) and there is a wall (according to MAP).
        """
        self.env.agent_pos = [7,1]
        _, reward, _, _ = self.env.step(action=4)
        self.assertEqual(reward, -6, 'Gridworld step failed, should get reward=-6 fpr stepping into wall')

    def test_within_bounds(self):
        """Intention: Test the step function for checking if a position is within bounds, try invlid position.

        Oracle: Asserts that False is returned when the value for row is negative.
        """
        within_bounds = self.env.within_bounds(column=0, row=-1)
        self.assertEqual(within_bounds, False, 'Gridworld within_bounds failed')

if __name__ == '__main__':
    unittest.main()

