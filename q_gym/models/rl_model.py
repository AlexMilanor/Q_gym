from typing import Callable

import numpy as np
import gym

GymEnv = gym.wrappers.time_limit.TimeLimit


class Transducer:
    """
    Translating layer between the GymEnv and the Q-learning table.

    Temporary. For now, it assumes specifically the CartPole-v1 model.
    """
    def get_idx_from_state(self, state):
        """
        Get the mesh index from that state
        """
        num_points = 100

        x_range = np.linspace(-2.4, 2.4, num_points)
        x_dot_range = np.linspace(-10, 10, num_points)
        w_range = np.linspace(-12 * np.pi / 180, 12 * np.pi / 180, num_points)
        w_dot_range = np.linspace(-10, 10, num_points)

        idx_x = np.argmin(np.abs(x_range - state[0]))
        idx_x_dot = np.argmin(np.abs(x_dot_range - state[1]))
        idx_w = np.argmin(np.abs(w_range - state[2]))
        idx_w_dot = np.argmin(np.abs(w_dot_range - state[3]))

        return (idx_x, idx_x_dot, idx_w, idx_w_dot)


class QLearn:
    """
    Tabular Q-learning model for reinforcement learning.
    It consists of a Q-table for deriving the optimal policies,
    learning parameters and methods for training, given the 
    OpenAI Gym environment. 

    Most of the code reference are the original implementations made for
    the MITx course 
    "Machine Learning with Python - From Linear Models to Deep Learning". 

    Parameters
    ----------
    state_dim : tuple
        Dimensions for the Q-table created.
    
    alpha : float
        Learning rate for the Q-value iteration. 
        Defaults to 0.1.

    gamma : float
        Discount factor of the utility function.
        Defaults to 0.5.

    train_epsilon : float
        Probability that the model will randomly explore the observable 
        space, instead of exploiting the Q-table, when training.
        Defaults to 0.5.

    run_epsilon : float 
        Probability that the model will randomly explore the observable 
        space, instead of exploiting the Q-table, when testing.
        Defaults to 0.05.
    """

    # ----------------------------------------------------------------------
    # Constructors

    def __init__(
        self, 
        state_dim:tuple,
        boilerplate:Transducer=None,
        alpha=0.1, 
        gamma=0.5, 
        train_epsilon=0.5, 
        run_epsilon=0.05
    ):

        self.alpha = alpha
        self.gamma = gamma
        self.train_epsilon = train_epsilon
        self.run_epsilon = run_epsilon
        self.boilerplate = boilerplate

        self.q_func = self._build_q_func(state_dim)


    def _build_q_func(self, dimensions:tuple):
        """Build the Q-table as a np.ndarray"""
        return np.zeros(dimensions)


    def q_table_update(
        self,
        q_func:np.ndarray,
        curr_state_idx:tuple,
        act_idx:int,
        reward:float,
        next_state_idx:tuple,
        terminal:bool
    ):
        """
        Update the Q-table for a given transition

        Parameters
        ----------
        q_func : np.ndarray 
            current Q-function
        curr_state: 
            two indices describing the current state
        act_idx: 
            index of the current action
        object_index (int): 
            index of the current object
        reward (float): 
            the immediate reward the agent receives from playing current command
        next_state_1, next_state_2 (int, int): 
            two indices describing the next state
        terminal (bool): 
            True if this episode is over

        Returns:
        updated_q_table : An updated version of the q_table
        """
        state_and_action = curr_state_idx + (action_index,)
    
        max_q_next_state = np.max(q_func[next_state_idx])
        q_current_state = q_func[state_and_action]

        if not terminal:
            new_q = ( 1.0 - ALPHA ) * q_current_state + ALPHA * ( reward + GAMMA * max_q_next_state)
        
        else:
            new_q = ( 1.0 - ALPHA ) * q_current_state + ALPHA * reward

        new_q_func = np.copy(q_func)
        new_q_func[state_and_action] = new_q

        return new_q_func


    def select_action(
        self,
        state_idx:tuple,
        train_mode:bool,
    ):
        """Returns the index number for the next action.

        The epsilon greedy strategy for taking the action depends on if the
        model is in training mode or in testing/running mode.
        """
        epsilon = self.train_epsilon if train_mode else self.run_epsilon
        return self.epsilon_greedy_strategy(state_idx, self.q_func, epsilon)


    def _epsilon_greedy_strategy(
        self,
        state_idx, 
        q_func, 
        epsilon
    ):
        """
        Decides on the next action, as selected by an epsilon-Greedy exploration
        policy.

        Parameters
        ----------
        state_idx : tuple 
            Tuple describing the current state
        q_func : np.ndarray 
            Current Q-table
        epsilon : float
            Probability of choosing a random action

        Returns:
        act_idx: int describing the index of the action to take
        """
        randomness = np.random.uniform(0,1)
        
        if randomness < epsilon:
            # Exploration
            act_idx = np.random.choice([0,1])
            
        else:
            # Exploitation
            current_state_q = q_func[state_idx]
            act_idx = np.argmax(current_state_q)

        return act_idx


    def run_episode(
        self,
        train_mode:bool,
        env:GymEnv):
        """
        Runs one episode of the gym environment.
        When in training mode, updates the Q-table,
        while when in running mode, computes and return the cumulative
        discounted reward.


        Parameters
        ----------
        train_mode : bool 
            Flag defining if we are in training mode or testing mode
        env : GymEnv 
            An object that is a gym environment, gotten from gym.make.
            It can be a non gym environment, as long as it is similar
            to one.
            This means it needs
            - a reset method, that starts the episode (and returns starting state) 
            - a state attribute, that gives the state at anytime
            - a step method, that receives an action and returns a tuple with
            4 elements, "observation" (object), object representing the state 
                        "reward" (float), reward of the previous action
                        "done" (boolean), flag defining if the episode ended
                        "info" (dict), dict with diagnostic information

        Returns:
            None
        """
        epi_reward = 0

        # initialize for each episode
        state = env.reset()
        state_idx = self.boilerplate.get_idx_from_state(state)
        terminal = False

        step_count = 0
        while not terminal:
            
            # Choose next action and execute
            act_idx = self.select_action(state_idx, train_mode)
            action = act_idx
            
            new_state, reward, terminal, info = env.step(action)
            new_state_idx = self.boilerplate.get_idx_from_state(new_state)

            if train_mode:
                # update Q table

                self.q_func = self.q_table_update(
                                    self.q_func,
                                    state_idx,
                                    act_idx,
                                    reward,
                                    new_state_idx,
                                    terminal)

            else:
                # update reward

                epi_reward += ( self.gamma ** step_count ) * reward
                
            # prepare next step
            state_idx = new_state_idx
            step_count += 1

            if step_count == 1000:
                # Wether the environment ends or not after a long time,
                # we will end it.
                terminal = True

        if not train_mode:
            print(f"Final reward : {epi_reward}")
            return epi_reward