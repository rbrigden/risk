import random
import unittest

import riskbot


class SimulatorTests(unittest.TestCase):
    def test_actions_and_invalid_input(self):
        self.assertEqual(riskbot.REGIONS, tuple(tuple(range(i, i + 7)) for i in range(0, 42, 7)))
        self.assertEqual(len(riskbot.DIRECTED_EDGES), 162)
        self.assertEqual(riskbot.ACTION_COUNT, 1016)
        self.assertEqual(riskbot.decode_action(41), ("place", 41, None, None))
        self.assertEqual(riskbot.decode_action(riskbot.encode_action("attack", 9, 40, 3)),
                         ("attack", 9, 40, 3))
        with self.assertRaises(ValueError):
            riskbot.encode_action("attack", 0, 35, 1)
        for action in range(riskbot.ACTION_COUNT):
            self.assertEqual(riskbot.encode_action(*riskbot.decode_action(action)), action)
        sim = riskbot.Simulator(seed=17)
        before = sim.observe(sim.active_player)
        for invalid in (-1, 2**100, "0", riskbot.ACTION_COUNT - 1):
            with self.assertRaises(ValueError):
                sim.step(invalid)
            self.assertEqual(sim.observe(sim.active_player), before)
        with self.assertRaises(ValueError):
            sim.observe(-1)
        with self.assertRaises(ValueError):
            riskbot.Simulator(num_players=7)

    def test_seeded_episodes(self):
        for players in range(2, 7):
            with self.subTest(players=players):
                sim = riskbot.Simulator(players, seed=123, max_turns=25)
                replay = riskbot.Simulator(players, seed=123, max_turns=25)
                rng = random.Random(456)
                obs = sim.observe(sim.active_player)
                self.assertEqual(obs, replay.observe(replay.active_player))
                spec = sim.observation_spec()
                while sim.active_player is not None:
                    for seat in range(players):
                        view = sim.observe(seat)
                        self.assertEqual(view["player"], seat)
                        self.assertEqual(any(view["action_mask"]), seat == sim.active_player)
                    self.assertEqual(list(obs), list(spec))
                    for key, (length, lower, upper) in spec.items():
                        values = (obs[key],) if length is None else obs[key]
                        self.assertEqual(len(values), 1 if length is None else length)
                        self.assertTrue(all(lower <= value <= upper for value in values))
                    legal = [i for i, allowed in enumerate(obs["action_mask"]) if allowed]
                    action = rng.choice(legal)
                    obs, rewards = sim.step(action)
                    self.assertEqual((obs, rewards), replay.step(action))
                self.assertFalse(any(obs["action_mask"]))
                self.assertNotEqual(obs["terminated"], obs["truncated"])
                with self.assertRaises(RuntimeError):
                    sim.step(0)
                self.assertEqual(sim.reset(123), replay.reset(123))

    def test_draw(self):
        sim = riskbot.Simulator(seed=12, max_turns=1)
        obs = sim.observe(sim.active_player)
        while obs["phase"] == 0:
            action = obs["action_mask"].index(True)
            obs, _ = sim.step(action)
        obs, _ = sim.step(riskbot.encode_action("finish_attack"))
        obs, rewards = sim.step(riskbot.encode_action("pass_fortify"))
        self.assertTrue(obs["truncated"])
        self.assertEqual(obs["winner"], -1)
        self.assertEqual(rewards, (0, 0))
