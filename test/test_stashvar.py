import pytest
import umpost.stashvar_cmip6 as stashvar


@pytest.mark.parametrize(
    "code,model,expected_long_name",
    [
        (3173, None, "QCF INCR: bdy layer negative"),
        (3173, stashvar.MODEL_ESM1PX, "soil (heterotrophic) respiration on tiles"),
        (3861, None, "UNKNOWN VARIABLE"),
        (3861, stashvar.MODEL_ESM1PX, "NITROGEN POOL PLANT - LEAF ON TILES"),
    ]
)
def test_get_stashinfo_overwrites(code, model, expected_long_name):
    """
    Test that model name overwrites are correctly extracted,
    and that defaults are relied on when no overwrite is specified,
    and when specified overwrite does not exist.
    """
    var = stashvar.get_stashinfo(code, model=model)
    assert var[0] == expected_long_name


def test_get_stashinfo_wrong_model():
    """
    Check that keyerror raised from unknown model key.
    """
    code = 2
    with pytest.raises(KeyError):
        stashvar.get_stashinfo(code, model="FAKE_MODEL")
