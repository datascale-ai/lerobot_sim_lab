#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from numpy import array


class PassiveViewerWrapper(gym.Wrapper):
    """Gym wrapper that opens a passive MuJoCo viewer automatically.

    The wrapper starts a MuJoCo viewer in passive mode as soon as the
    environment is created so the user no longer needs to use
    ``mujoco.viewer.launch_passive`` or any context–manager boiler-plate.

    The viewer is kept in sync after every ``reset`` and ``step`` call and is
    closed automatically when the environment itself is closed or deleted.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        show_left_ui: bool = False,
        show_right_ui: bool = False,
        default_camera: str = "camera_front_new",
        lock_camera: bool = False,
        sync_every_n_steps: int = 1,
    ) -> None:
        """
        Args:
            env: The environment to wrap
            show_left_ui: Show left UI panel (not implemented in passive viewer)
            show_right_ui: Show right UI panel (not implemented in passive viewer)
            default_camera: Name of the camera to use as initial view
            lock_camera: If True, lock to the specified camera (cannot adjust with mouse).
                        If False, use the camera as initial view but allow free adjustment.
            sync_every_n_steps: Sync viewer every N steps (default 1 = every step).
                               Set to higher values (e.g., 5-10) for faster visualization.
        """
        super().__init__(env)
        self._sync_every_n_steps = sync_every_n_steps
        self._step_count = 0

        # Launch the interactive viewer.  We expose *model* and *data* from the
        # *unwrapped* environment to make sure we operate on the base MuJoCo
        # objects even if other wrappers have been applied before this one.
        self._viewer = mujoco.viewer.launch_passive(
            env.unwrapped.model,
            env.unwrapped.data,
            # show_left_ui=show_left_ui,
            # show_right_ui=show_right_ui,
        )
        
        # Set default camera view
        if default_camera:
            try:
                camera_id = mujoco.mj_name2id(
                    env.unwrapped.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    default_camera
                )
                if camera_id >= 0:
                    if lock_camera:
                        # Lock to the specified camera (no mouse adjustment)
                        self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        self._viewer.cam.fixedcamid = camera_id
                        print(f"📷 Camera locked to '{default_camera}' (use bracket keys [ ] to switch)")
                    else:
                        # Set free camera to match the specified camera's view
                        # This allows mouse adjustment while starting from the desired angle
                        
                        # Get camera position and target from XML definition
                        cam_pos = array([0.   , -0.45 , 0.1])
                        # For camera_front_new: pos="0 0.65 0.254" looking at lookat_target="0 0.08 0.15"
                        # Set the lookat point (where camera is pointing)
                        self._viewer.cam.lookat[:] = [0.0, 0.08, 0.15]  # lookat_target position
                        
                        # Calculate distance from camera to lookat point
                        cam_to_target = cam_pos - self._viewer.cam.lookat
                        distance = float(np.linalg.norm(cam_to_target))
                        self._viewer.cam.distance = distance
                        
                        # Calculate azimuth and elevation
                        # azimuth: angle in XY plane (0 = +X, 90 = +Y, 180 = -X, 270 = -Y)
                        # elevation: angle from XY plane (-90 = bottom, 0 = level, 90 = top)
                        azimuth = float(np.degrees(np.arctan2(cam_to_target[1], cam_to_target[0])))
                        xy_dist = float(np.sqrt(cam_to_target[0]**2 + cam_to_target[1]**2))
                        elevation = float(np.degrees(np.arctan2(cam_to_target[2], xy_dist)))
                        
                        self._viewer.cam.azimuth = azimuth
                        self._viewer.cam.elevation = elevation
                        
                        # Ensure free camera mode
                        self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                        
                        print(f"📷 Starting with '{default_camera}' view (free to adjust with mouse)")
            except Exception as e:
                print(f"Warning: Could not set default camera '{default_camera}': {e}")

        # Make sure the first frame is rendered.
        self._viewer.sync()

    # ---------------------------------------------------------------------
    # Gym API overrides

    def reset(self, **kwargs):  # type: ignore[override]
        observation, info = self.env.reset(**kwargs)
        self._step_count = 0  # Reset step counter for new episode
        self._viewer.sync()
        return observation, info

    def step(self, action):  # type: ignore[override]
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        # Only sync every N steps for faster visualization
        if self._step_count % self._sync_every_n_steps == 0:
            self._viewer.sync()
        return observation, reward, terminated, truncated, info

    def close(self) -> None:  # type: ignore[override]
        """Close both the passive viewer and the underlying gym environment.

        MuJoCo's `Renderer` gained a `close()` method only in recent versions
        (>= 2.3.0).  When running with an older MuJoCo build the renderer
        instance stored inside `env.unwrapped._viewer` does not provide this
        method which causes `AttributeError` when the environment is closed.

        To remain version-agnostic we:
          1. Manually dispose of the underlying viewer *only* if it exposes a
             `close` method.
          2. Remove the reference from the environment so that a subsequent
             call to `env.close()` will not fail.
          3. Close our own passive viewer handle.
          4. Finally forward the `close()` call to the wrapped environment so
             that any other resources are released.
        """

        # 1. Tidy up the renderer managed by the wrapped environment (if any).
        base_env = self.env.unwrapped  # type: ignore[attr-defined]
        if hasattr(base_env, "_viewer"):
            viewer = base_env._viewer
            if viewer is not None and hasattr(viewer, "close") and callable(viewer.close):
                try:  # noqa: SIM105
                    viewer.close()
                except Exception:
                    # Ignore errors coming from older MuJoCo versions or
                    # already-freed contexts.
                    pass
            # Prevent the underlying env from trying to close it again.
            base_env._viewer = None

        # 2. Close the passive viewer launched by this wrapper.
        try:  # noqa: SIM105
            self._viewer.close()
        except Exception:  # pragma: no cover
            # Defensive: avoid propagating viewer shutdown errors.
            pass

        # 3. Let the wrapped environment perform its own cleanup.
        self.env.close()

    def __del__(self):
        # "close" may raise if called during interpreter shutdown; guard just
        # in case.
        if hasattr(self, "_viewer"):
            try:  # noqa: SIM105
                self._viewer.close()
            except Exception:
                pass
