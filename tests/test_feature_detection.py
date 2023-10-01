import pytest
import core.img_proccessing as imp
import utils

MIN_CNT_FEATURES = 800

img_list = utils.get_image_files('./tests/images/')

@pytest.mark.parametrize("img_name", img_list )
def test_detect_feature( img_name ):
    desc, desc_type = imp.get_image_data( img_name, MIN_CNT_FEATURES )

    assert desc.shape[0] >= MIN_CNT_FEATURES

@pytest.mark.parametrize("img_name", img_list )
def test_detect_feature_with_filters( img_name ):
    filter_flags = imp.FilterFlags.DENOISE.value | \
                   imp.FilterFlags.GAUSSIAN.value | \
                   imp.FilterFlags.RES_ENHANCEMENT.value | \
                   imp.FilterFlags.HIST_EQUALIZATION.value

    desc, desc_type = imp.get_image_data_v2(img_name,
                                            MIN_CNT_FEATURES, 
                                            filter_flags)

    assert desc.shape[0] >= MIN_CNT_FEATURES