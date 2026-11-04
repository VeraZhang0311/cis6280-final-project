"""Evaluator-only contact/support evidence; these fields never enter the planner."""
from collections import deque
from pathlib import Path

import numpy as np

from assignment.training.planning.diagnostics import write_json


class TailEvidence:
    def __init__(self, window=56):
        if window < 1:
            raise ValueError('tail evidence window must be positive')
        self.rows = deque(maxlen=window)

    def capture(self, task_env, step):
        import mujoco
        from assignment.training.data.signals import in_goal_zone, structure, tower
        env = task_env.env
        row = env.history[-1]
        rooted, _ = structure(row, task_env.goal, task_env.half)
        member, _ = tower(row, task_env.goal, task_env.half)
        lookup = {g: i for i, g in enumerate(env.object_geoms)}
        contacts = []
        for ci, c in enumerate(env.data.contact):
            if c.geom1 not in lookup and c.geom2 not in lookup:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(env.model, env.data, ci, force)
            contacts.append(dict(geom1=env.model.geom(c.geom1).name, geom2=env.model.geom(c.geom2).name,
                                 object1=lookup.get(c.geom1), object2=lookup.get(c.geom2),
                                 distance_m=float(c.dist), normal_z=float(c.frame[2]),
                                 compression_n=float(force[0])))
        self.rows.append(dict(control_step=step, sim_time=step*.1, count=int(member.sum()),
                              goal_xy=list(task_env.goal), goal_half_width=task_env.half,
                              in_zone=in_goal_zone(row['object_bounds'], task_env.goal, task_env.half).tolist(),
                              structure_members=rooted.tolist(), tower_members=member.tolist(),
                              contacts=contacts, **{k: v.tolist() for k, v in row.items()}))

    def save(self, folder, task_env):
        folder = Path(folder)
        objects = [dict(index=i, material=obj.material, size=list(obj.size))
                   for i, obj in enumerate(task_env.env.config.objects)]
        write_json(folder/'tail-state.json', list(self.rows))
        write_json(folder/'objects.json', objects)


def save_keyframes(folder, frames, control_steps, count):
    import imageio.v2 as imageio
    folder = Path(folder)/'keyframes'
    folder.mkdir(parents=True, exist_ok=True)
    for step, frame in list(zip(control_steps, frames))[-count:]:
        imageio.imwrite(folder/f'step-{step}.png', frame)
