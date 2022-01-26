import gym
import numpy as np

from q_gym.models.rl_model import QLearn, Transducer

class TestTransducer:
    def test_get_idx_from_state(self):
        #BUILD   ------
        transducer = Transducer()   

        #OPERATE ------
        state = np.array([-2.38, -9.9, (-12 * np.pi / 180)*(1+1/100), -9.9])
        state_idx = transducer.get_idx_from_state(state)

        #CHECK   ------
        np.testing.assert_array_equal(state_idx, [0, 0, 0, 0])


class TestQLearn:
    def test_q_func_build(self):
        #BUILD   ------
        model = QLearn((3, 2, 5))

        #OPERATE ------
        Q = model._build_q_func((3,2,5))
        res =[[[0., 0., 0., 0., 0.]
              ,[0., 0., 0., 0., 0.]],

              [[0., 0., 0., 0., 0.]
              ,[0., 0., 0., 0., 0.]],

              [[0., 0., 0., 0., 0.]
              ,[0., 0., 0., 0., 0.]]]


        #CHECK   ------
        np.testing.assert_array_equal(Q, res)


    def test_q_func_build_att(self):
        #BUILD   ------
        model = QLearn((3, 2, 5))

        #OPERATE ------
        Q = model.q_func
        res =[[[0., 0., 0., 0., 0.]
              ,[0., 0., 0., 0., 0.]],

              [[0., 0., 0., 0., 0.]
              ,[0., 0., 0., 0., 0.]],

              [[0., 0., 0., 0., 0.]
              ,[0., 0., 0., 0., 0.]]]


        #CHECK   ------
        np.testing.assert_array_equal(Q, res)





