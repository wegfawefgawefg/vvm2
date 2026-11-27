"""Two small physical worlds in torch, batched, for old/control7.py. No outside package.

A world holds n episodes at once. `obs()` is what the driver may show the machine, `step`
takes one action an episode, `score` is the result so far, `over()` says when to stop.
`greedy(two)` turns the first two numbers of the output slot into an action.
"""
import math

import torch


class CartPole:
    """A cart on a track with a pole hinged on top; push the cart left or right to keep the
    pole up. The standard equations (Barto, Sutton and Anderson 1983, as in the usual
    CartPole-v1): gravity 9.8, cart mass 1.0, pole mass 0.1, pole half-length 0.5, push
    10 newtons either way, a world step of 0.02 s by Euler's rule. An episode starts with
    all four state numbers uniform in +-0.05 and ends when the cart is more than 2.4 from
    the middle or the pole more than 12 degrees from upright. The score is the number of
    steps balanced, capped at 500.

    State: cart position, cart velocity, pole angle, pole angular velocity. With
    `hide=True` only the two positions are shown."""
    cap = 500

    def __init__(self, n, g, hide=False):
        self.s = (torch.rand(n, 4, generator=g) - 0.5) * 0.1
        self.alive = torch.ones(n, dtype=torch.bool)
        self.score = torch.zeros(n)
        self.hide, self.t = hide, 0
        self.n_obs = 2 if hide else 4

    def obs(self):
        return self.s[:, [0, 2]] if self.hide else self.s

    @staticmethod
    def greedy(two):
        """Push right when the first number is the larger."""
        return two[:, 0] > two[:, 1]

    def step(self, right):
        x, xd, th, thd = self.s.unbind(1)
        force = torch.where(right, 10.0, -10.0)
        cos, sin = torch.cos(th), torch.sin(th)
        temp = (force + 0.1 * 0.5 * thd * thd * sin) / 1.1
        thacc = (9.8 * sin - cos * temp) / (0.5 * (4.0 / 3.0 - 0.1 * cos * cos / 1.1))
        xacc = temp - 0.1 * 0.5 * thacc * cos / 1.1
        new = torch.stack([x + 0.02 * xd, xd + 0.02 * xacc, th + 0.02 * thd, thd + 0.02 * thacc], 1)
        self.score += self.alive.float()            # the step that drops it counts, as usual
        self.s = torch.where(self.alive[:, None], new, self.s)
        ok = (self.s[:, 0].abs() <= 2.4) & (self.s[:, 2].abs() <= 12 * math.pi / 180)
        self.alive = self.alive & ok
        self.t += 1

    def over(self):
        return self.t >= self.cap or not bool(self.alive.any())


class Pendulum:
    """Swing a hanging pendulum up and hold it there with a weak motor (the usual
    Pendulum-v1): gravity 10, mass 1, length 1, torque at most 2 either way (too weak to
    lift it straight up), a world step of 0.05 s, speed held to +-8. An episode is 200 steps
    from a random angle and a speed in +-1. Each step costs angle^2 + 0.1 speed^2 +
    0.001 torque^2 (angle 0 is upright), and the score is minus the total: about -1200 for
    doing nothing useful, about -150 for a good swing-up. `up` counts how many of the last
    50 steps were within 0.3 radians of upright.

    Shown: cos and sin of the angle, and the speed."""
    cap = 200
    n_obs = 3

    def __init__(self, n, g, hide=False):
        self.th = (torch.rand(n, generator=g) * 2 - 1) * math.pi
        self.thd = torch.rand(n, generator=g) * 2 - 1
        self.score, self.up, self.t = torch.zeros(n), torch.zeros(n), 0
        self.alive = torch.ones(n, dtype=torch.bool)

    def obs(self):
        return torch.stack([torch.cos(self.th), torch.sin(self.th), self.thd], dim=1)

    @staticmethod
    def greedy(two):
        """Torque: twice the difference of the two numbers, held to +-2."""
        return 2 * (two[:, 0] - two[:, 1]).clamp(-1, 1)

    def step(self, u):
        u = u.clamp(-2, 2)
        angle = (self.th + math.pi) % (2 * math.pi) - math.pi
        self.score -= angle ** 2 + 0.1 * self.thd ** 2 + 0.001 * u ** 2
        if self.t >= self.cap - 50:
            self.up += (angle.abs() < 0.3).float()
        self.thd = (self.thd + (15.0 * torch.sin(self.th) + 3.0 * u) * 0.05).clamp(-8, 8)
        self.th = self.th + self.thd * 0.05
        self.t += 1

    def over(self):
        return self.t >= self.cap
