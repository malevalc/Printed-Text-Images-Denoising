import sys
import cv2
import globals
import math
import numpy as np
import os
import random
import scipy.ndimage as sci
from skimage import transform as tf


def size_handle(img):
    """ Resizes and reshapes the input image.

        Resize the image to 512, 256 and reshape it to fit the model input shape.

        # Arguments
            img: input image
        # Returns
            output image
    """
    # set all images to size 512, 256
    output = cv2.resize((img[:] / 255.0), (globals.INPUT_COLS, globals.INPUT_ROWS))
    return output.reshape(1, globals.INPUT_ROWS, globals.INPUT_COLS, 1)


def load_dataset(datatype, file_list, folder):
    """ Loads the dataset depending on its type.

        Load all images, store them in a list, then normalize the x set.
        Displays user friendly information on computation.

        # Arguments
            datatype: TRAIN/TEST
            file_list: list of all images to store
            folder: which damaged folder (could be train or test)
        # Returns
            the image list(s)
    """
    print('Loading dataset...')

    # load the data and store in an array
    load_damaged = size_handle(cv2.imread(os.path.join(folder, file_list[0]), cv2.IMREAD_GRAYSCALE))
    load_cleaned = size_handle(cv2.imread(os.path.join(globals.TRAIN_CLEANED, file_list[0]), cv2.IMREAD_GRAYSCALE)) if datatype is globals.TRAIN else []
    sys.stdout.write("|#")
    for file_index, file_name in enumerate(file_list[1:]):
        load_damaged = np.append(load_damaged, size_handle(cv2.imread(os.path.join(folder, file_name), cv2.IMREAD_GRAYSCALE)), axis=0)
        load_cleaned = np.append(load_cleaned, size_handle(cv2.imread(os.path.join(globals.TRAIN_CLEANED, file_name), cv2.IMREAD_GRAYSCALE)),
                                 axis=0) if datatype is globals.TRAIN else []
        sys.stdout.write("#")
        if (file_index+2) % 100 == 0:
            sys.stdout.write("\n")
        if (file_index+2) % 10 == 0:
            sys.stdout.write("|")
    sys.stdout.write('\nDone\n')
    load_damaged[:] = [img - 0.5 for img in load_damaged]
    return (load_damaged, load_cleaned) if datatype is globals.TRAIN else load_damaged


def augment_dataset(augment_list,
                    augmentation_amount=15,
                    max_rotation_angle=30,
                    max_shift_ratio=0.2,
                    max_shear_ratio=0.1,
                    max_zoom_factor=0.25):
    """ Performs data augmentation on given dataset.

        Performs random data augmentation n times.
        Techniques include rotation, shifting, shearing, flipping and zooming.
        Displays user friendly information on computation.
        Saves augmented images to dataset folders.

        # Arguments
            augment_list: list of images to augment
            augmentation_amount: number of random augmentations to perform per image
            max_rotation_angle: maximum angle (degrees) by which perform rotation
            max_shift_ratio: maximum ratio of the image by which perform shifting
            max_shear_ratio: maximum ratio of the image by which perform shearing
            max_zoom_factor: maximum ratio of the image by which perform zooming
        # Returns
            None
    """
    print('Augmenting dataset...')
    progress = ''
    for files in range(len(augment_list)):
        for i in range(augmentation_amount):
            # load both files
            damaged_augmented_file = cv2.imread(os.path.join(globals.TRAIN_DAMAGED, augment_list[files]))
            cleaned_augmented_file = cv2.imread(os.path.join(globals.TRAIN_CLEANED, augment_list[files]))
            rows, cols, _ = damaged_augmented_file.shape

            # create random number factors
            rotate_random = random.random()
            width_shift_random = random.random()
            height_shift_random = random.random()
            shear_random = random.random()
            horizontal_flip_random = random.random()
            vertical_flip_random = random.random()
            zoom_random = random.random()

            # rotate both files by the first random factor
            damaged_augmented_file = sci.rotate(damaged_augmented_file, int(round(rotate_random * max_rotation_angle)),
                                                reshape=False, mode='nearest')
            cleaned_augmented_file = sci.rotate(cleaned_augmented_file, int(round(rotate_random * max_rotation_angle)),
                                                reshape=False, mode='nearest')

            # shift both files by random factors
            damaged_augmented_file = np.pad(damaged_augmented_file, ((0, 0), (int(math.ceil(cols * width_shift_random * max_shift_ratio)), 0), (0, 0)),
                                            mode='edge')[:, :-int(math.ceil(cols * width_shift_random * max_shift_ratio))]
            cleaned_augmented_file = np.pad(cleaned_augmented_file, ((0, 0), (int(math.ceil(cols * width_shift_random * max_shift_ratio)), 0), (0, 0)),
                                            mode='edge')[:, :-int(math.ceil(cols * width_shift_random * max_shift_ratio))]
            damaged_augmented_file = np.pad(damaged_augmented_file, ((int(math.ceil(rows * height_shift_random * max_shift_ratio)), 0), (0, 0), (0, 0)),
                                            mode='edge')[:-int(math.ceil(rows * height_shift_random * max_shift_ratio)), :]
            cleaned_augmented_file = np.pad(cleaned_augmented_file, ((int(math.ceil(rows * height_shift_random * max_shift_ratio)), 0), (0, 0), (0, 0)),
                                            mode='edge')[:-int(math.ceil(rows * height_shift_random * max_shift_ratio)), :]

            # perform shear on both files by random factor
            try:
                damaged_augmented_file = tf.warp(damaged_augmented_file,
                                                 inverse_map=tf.AffineTransform(shear=(shear_random - 0.5) * 2 * max_shear_ratio), mode='edge')
                cleaned_augmented_file = tf.warp(cleaned_augmented_file,
                                                 inverse_map=tf.AffineTransform(shear=(shear_random - 0.5) * 2 * max_shear_ratio), mode='edge')
            except ValueError:
                pass

            # flip both files (binary probability)
            damaged_augmented_file = np.fliplr(damaged_augmented_file) if horizontal_flip_random > 0.5 \
                else damaged_augmented_file
            damaged_augmented_file = np.flipud(damaged_augmented_file) if vertical_flip_random > 0.5 \
                else damaged_augmented_file
            cleaned_augmented_file = np.fliplr(cleaned_augmented_file) if horizontal_flip_random > 0.5 \
                else cleaned_augmented_file
            cleaned_augmented_file = np.flipud(cleaned_augmented_file) if vertical_flip_random > 0.5 \
                else cleaned_augmented_file

            # zoom in on both files by random factor
            damaged_augmented_file = cv2.resize(damaged_augmented_file,
                                                None, fx=8*max_zoom_factor*zoom_random+1,
                                                fy=8*max_zoom_factor*zoom_random+1, interpolation=cv2.INTER_LINEAR)
            damaged_augmented_file = damaged_augmented_file[(damaged_augmented_file.shape[0]-rows)//2:rows+(damaged_augmented_file.shape[0]-rows)//2,
                                                            (damaged_augmented_file.shape[1]-cols)//2:cols+(damaged_augmented_file.shape[1]-cols)//2]
            cleaned_augmented_file = cv2.resize(cleaned_augmented_file,
                                                None, fx=8*max_zoom_factor*zoom_random+1,
                                                fy=8*max_zoom_factor*zoom_random+1, interpolation=cv2.INTER_LINEAR)
            cleaned_augmented_file = cleaned_augmented_file[(cleaned_augmented_file.shape[0]-rows)//2:rows+(cleaned_augmented_file.shape[0]-rows)//2,
                                                            (cleaned_augmented_file.shape[1]-cols)//2:cols+(cleaned_augmented_file.shape[1]-cols)//2]

            # save both files to directory
            cv2.imwrite(os.path.join(globals.TRAIN_DAMAGED, str(augment_list[files][:-4]) + globals.AUG_KEY + str(i + 1) + '.png'),
                        np.asarray(damaged_augmented_file * 255.0, dtype=np.uint8))
            cv2.imwrite(os.path.join(globals.TRAIN_CLEANED, str(augment_list[files][:-4]) + globals.AUG_KEY + str(i + 1) + '.png'),
                        np.asarray(cleaned_augmented_file * 255.0, dtype=np.uint8))

            sys.stdout.write(".")

        if files % 10 == 0:
            progress += '|'
        progress += '#'
        sys.stdout.write("\r"+progress)
    sys.stdout.write(' done\n')


def delete_augment_dataset():
    """ Deletes the augmented dataset.

        Finds all files that contain the keyword 'augm' and deletes them.

        # Arguments
            None
        # Returns
            None
    """
    # find all files that contain keyword and delete them
    all_images = [file for file in os.listdir(globals.TRAIN_DAMAGED) if file.endswith('png')]
    images_delete = [delete for delete in all_images if globals.AUG_KEY in delete]
    for image_delete in range(len(images_delete)):
        os.remove(os.path.join(globals.TRAIN_DAMAGED, images_delete[image_delete]))
        os.remove(os.path.join(globals.TRAIN_CLEANED, images_delete[image_delete]))


def test_on_dataset(reference_folder, reference_model, time):
    """ Performs test on data.

        Loads the test set, predicts the cleaning process for all files in the set
        and store in folder.
        Displays user friendly information on computation.

        # Arguments
            reference_folder: folder where test files are stored
            reference_model: model created and trained to use to perform predictions
            time: start time of computation, for folder creation
        # Returns
            None
    """
    # make predictions and reformat all files
    test_list = [file for file in os.listdir(reference_folder)]
    test_images = load_dataset(globals.TEST, test_list, reference_folder)
    folder = os.path.join(globals.DIR, globals.OUTPUTS, time, globals.TEST_CLEANED)
    os.makedirs(folder)
    # make all predictions
    print('Making predictions...')
    for x in range(len(test_images)):
        file_name = test_images[x].reshape(1, globals.INPUT_ROWS, globals.INPUT_COLS, 1)
        prediction = reference_model.predict(file_name)
        prediction = prediction.reshape(globals.INPUT_ROWS, globals.INPUT_COLS)
        # prediction = np.clip(prediction, 0, 1)
        output_image = np.asarray(prediction * 255.0, dtype=np.uint8)
        cv2.imwrite((os.path.join(folder, test_list[x])), output_image)
        sys.stdout.write("#")
    sys.stdout.write(' done\n')
