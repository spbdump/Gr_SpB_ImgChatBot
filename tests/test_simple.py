import pytest
import utils

    

img_list = ["aa", "bb", "cc", "ca", "cb", "ad", "db"]
f_list = utils.get_random_images(img_list, 3)

# @pytest.fixture(params=f_list)
# def img_name(request):
#     return request.param

@pytest.mark.parametrize("img_name", f_list )
def test_check_cc(prefix_path, img_name):
    print(prefix_path)
    assert img_name == "cc"