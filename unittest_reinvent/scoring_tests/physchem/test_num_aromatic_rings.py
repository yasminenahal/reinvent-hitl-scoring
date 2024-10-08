import numpy as np
import numpy.testing as npt

from reinvent_hitl_scoring.scoring.enums import TransformationParametersEnum

from unittest_reinvent.fixtures.test_data import BENZENE, ANILINE, BENZO_A_PYRENE, DECALIN, PACLITAXEL
from unittest_reinvent.scoring_tests.physchem.base_setup import BaseSetup


class TestNumAromaticRingsScoreWithDoubleSigmoid(BaseSetup):

    def setUp(self):
        super().setup_attrs()
        specific_parameters = {
            self.csp_enum.TRANSFORMATION: {
                TransformationParametersEnum.LOW: 1,
                TransformationParametersEnum.HIGH: 3,
                TransformationParametersEnum.TRANSFORMATION_TYPE: self.tt_enum.STEP
            }
        }
        super().init(self.sf_enum.NUM_AROMATIC_RINGS, specific_parameters)
        super().setUp()

    def test_num_aromatic_rings_1(self):
        smiles = [BENZENE, ANILINE, BENZO_A_PYRENE, DECALIN, PACLITAXEL]
        values = np.array([1., 1., 0., 0., 1.])
        score = self.sf_state.get_final_score(smiles=smiles)
        npt.assert_array_almost_equal(score.total_score, values, 2)
