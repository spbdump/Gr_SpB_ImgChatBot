import cv2
import img_proccessing

def main():

    print("Check base functionality")
    path = './test_images/'
    img2 = cv2.imread(path+'photo_2023-02-09_02-17-55.jpg')
    img1 = cv2.imread(path+'photo_2023-02-09_02-17-55_copy.jpg')
    img3 = cv2.imread(path+'photo_2023-02-08_19-24-07.jpg ')
    
    # Check if the image was loaded successfully
    if img2 is None:
        print("Error: Could not load the image.")
        return

    print("Is same imgs: ", img_proccessing.compare_images_sift(img1, img2) )
    print("Is same imgs: ", img_proccessing.compare_images_sift(img1, img3) )

    # Show the image
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
