import numpy as np
from gymnasium import spaces 

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.trajectory_generator import Waypoints_Generator
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class Waypoints(BaseRLAviary):
    """Single Agent RL Agent: Track Waypoints"""
    
    def __init__(self,
                    drone_model: DroneModel=DroneModel.CF2X,
                    initial_xyzs=None,
                    initial_rpys=None,
                    physics: Physics=Physics.PYB,
                    pyb_freq: int = 240,
                    ctrl_freq: int = 30,
                    gui=False,
                    record=False,
                    obs: ObservationType=ObservationType.KIN,
                    act: ActionType=ActionType.RPM
                    ):
            """Initialization of a single agent RL environment.

            Using the generic single agent RL superclass.

            Parameters
            ----------
            drone_model : DroneModel, optional
                The desired drone type (detailed in an .urdf file in folder `assets`).
            initial_xyzs: ndarray | None, optional
                (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
            initial_rpys: ndarray | None, optional
                (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
            physics : Physics, optional
                The desired implementation of PyBullet physics/custom dynamics.
            pyb_freq : int, optional
                The frequency at which PyBullet steps (a multiple of ctrl_freq).
            ctrl_freq : int, optional
                The frequency at which the environment steps.
            gui : bool, optional
                Whether to use PyBullet's GUI.
            record : bool, optional
                Whether to save a video of the simulation.
            obs : ObservationType, optional
                The type of observation space (kinematic information or vision)
            act : ActionType, optional
                The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

            """
            waypoint_gen = Waypoints_Generator()
            self.waypoints = waypoint_gen.spiral()
            self.wp_cnt = 0
            self.EPISODE_LEN_SEC = 8


            super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )


    ################################################################################
    
    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        
        ############################################################
        #### OBS SPACE OF SIZE 21
        #### Observation vector ### X Y Z R P Y VX VY VZ WX WY WZ X1 Y1 Z1 X2 Y2 Z2 X3 Y3 Z3

        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([lo,lo,0,lo,lo,lo,lo,lo,lo,lo,lo,lo,lo,lo,0,lo,lo,0,lo,lo,0])
        obs_upper_bound = np.array([hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi])

        #### Add action buffer to observation space ################
        act_lo = -1
        act_hi = +1
        for i in range(self.ACTION_BUFFER_SIZE):
            if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
            elif self.ACT_TYPE==ActionType.PID:
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
            elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo] for i in range(self.NUM_DRONES)])])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi] for i in range(self.NUM_DRONES)])])

        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
            
        #### OBS SPACE OF SIZE 21
        obs_21 = np.zeros((self.NUM_DRONES,21))
        for i in range(self.NUM_DRONES):
            obs = self._getDroneStateVector(i)
            obs_21[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16], self.waypoints[self.wp_cnt], self.waypoints[self.wp_cnt + 1], self.waypoints[self.wp_cnt + 2]]).reshape(21,) 
        ret = np.array([obs_21[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
        #### Add action buffer to observation #######################
        for i in range(self.ACTION_BUFFER_SIZE):
            ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
        return ret

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        ret = max(0, 2 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)
        return ret

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0001:
            return True
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
    
    #################################################################################