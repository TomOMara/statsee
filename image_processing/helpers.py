import cv2


def clear_tmp_on_run():
    import os
    import glob
    files = glob.glob(os.getcwd() + '/tmp/*')
    for f in files:
        os.remove(f)


def show_image(image):
    if type(image) is str:
        image = cv2.imread(image)

    cv2.imshow("output", image)
    cv2.waitKey(0)



def convert_mask_to_3D_image(mask):
    # If image is already 3d
    if array_is_3D(mask):
        return mask
    else:
        # otherwise write image and re-read it.
        cv2.imwrite("images/tmp.png", mask)
        mask_as_image = cv2.imread("images/tmp.png")

        return mask_as_image


def array_is_3D(image):
    return len(image.shape) == 3