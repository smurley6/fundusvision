import numpy as np
import pytest

from fundusvision.data.odir import keyword_to_label


@pytest.mark.parametrize(
    ("keywords", "expected"),
    [
        ("normal fundus", 0),
        ("Normal Fundus ", 0),
        ("normal fundus，lens dust", 0),  # full-width comma + artefact token
        ("lens dust, normal fundus", 0),
        ("cataract", 1),
        ("normal fundus，cataract", 1),  # any pathology makes the eye abnormal
        ("moderate non proliferative retinopathy", 1),
        ("lens dust", 1),  # artefact only: no evidence of a normal fundus
        ("", 1),
        (None, 1),
        (np.nan, 1),
    ],
)
def test_keyword_to_label(keywords, expected):
    assert keyword_to_label(keywords) == expected
