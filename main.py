import architecture as arch
import data
from datetime import datetime
import globals
from keras import optimizers
import os
import random
import sys
import visualization as vis
from keras.utils import plot_model


class Console(object):
    """ Class created to take care of the std output
        It allows printing the output both to console and to the defined text file.
    """
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def ask_parameters():
    """ Make the user (trainer) input his hyper-parameters.

    It makes the user set the parameters, which can be numbers or a choice
    (then choice between values, with setting of limits)

    # Arguments
        None
    # Returns
        a Parameter object created from the input parameters
    """
    # Ask user to specify hyper-parameters for the compilation
    main_model = int(input("Choose a model (1. Simple, 2. VGG, 3. RESNET, 4. SQUEEZENET): "))
    while main_model < 1 or main_model > 3:
        main_model = int(input("Choose a model (1. Simple, 2. VGG, 3. RESNET, 4. SQUEEZENET): "))
    main_model = 'simple' if main_model == 1 else globals.VGG if main_model == 2 else globals.RESNET if main_model == 3 else globals.SQUEEZENET
    main_batch_size = int(input("Set a batch size: "))
    main_epochs = int(input("Set a number of epochs: "))
    main_optimizer = int(input("Choose an optimizer (1. SGD with Nesterov Momentum, 2. Adam): "))
    while main_optimizer < 1 or main_optimizer > 2:
        main_optimizer = int(input("Choose an optimizer (1. SGD with Nesterov Momentum, 2. Adam): "))
    main_lr = float(input("Set the learning rate value for the optimizer (0.01, 0.001 etc): "))
    main_data = input("Do you want to apply data augmentation (Y/N)? ")
    while main_data != 'Y' and main_data != 'N' and main_data != 'y' and main_data != 'n':
        main_data = input("Do you want to apply data augmentation (Y/N)? ")
    main_data = globals.YES if (main_data == 'Y' or main_data == 'y') else globals.NO

    return vis.Parameters(main_batch_size, main_epochs, main_optimizer, main_lr, main_data, main_model)


def compile_whole(comp_parameters=vis.Parameters(50, 10, globals.SGD, 0.01, globals.NO)):
    """ Compiles the whole model with the given parameters

    Folder creation for results saving. Dataset obtained, then augmented if set so.
    Optimizer then created with given parameters, then used to create and compile
    the model. Model architecture is displayed. Dataset (raw or augmented) is obtained,
    then used to train the chosen model. Training takes places while respecting the
    parameters (epochs, batch size, optimizer and learning rate). The model is then
    saved for later use (e.g. future predictions). Testing is performed using the test
    dataset. Then visualization is exported regarding the model (i.e. parameters are
    saved in a csv file, accuracy and loss are plotted, first layer filters are
    displayed and computation total time is printed.
    Every output is then zipped and sent by email.

    # Arguments
        Parameter object, making the parameters
    # Returns
        None
    """
    # CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # pydot/graphviz path
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    model_main = 0
    start_time = datetime.now()

    compute_time, target_folder = vis.create_folder()

    output = open(target_folder+globals.STD_OUTPUT_FILE, 'w')
    sys.stdout = Console(sys.stdout, output)

    # get all the training images, augment them if stated so, then shuffle the dataset
    reference_images = [file for file in os.listdir(globals.TRAIN_DAMAGED) if file.endswith('png')]
    if comp_parameters.augmentation:
        data.augment_dataset(reference_images, augmentation_amount=1)

    # call the function to create and compile the model depending one the type of optimizer
    if comp_parameters.optimizer is globals.SGD:
        model_main = arch.create_compile_model(globals.INPUT_DIM, model_type=comp_parameters.model,
                                               model_optimizer=optimizers.SGD(lr=comp_parameters.learning_rate, decay=0.001, momentum=0.9, nesterov=True))
    elif comp_parameters.optimizer is globals.ADAM:
        model_main = arch.create_compile_model(globals.INPUT_DIM, model_type=comp_parameters.model,
                                               model_optimizer=optimizers.Adam(lr=comp_parameters.learning_rate, decay=0.001))

    # plot model architecture
    plot_model(model_main, to_file=comp_parameters.model+'.png', show_shapes=True)
    # display the model architecture
    model_main.summary()

    file_list = [file for file in os.listdir(globals.TRAIN_DAMAGED) if file.endswith('png')]
    random.shuffle(file_list)

    # obtain the data to train the model
    train_damaged, train_cleaned = data.load_dataset(globals.TRAIN, file_list, globals.TRAIN_DAMAGED)

    # fit the model using hyper-parameters defined
    model_history = model_main.fit(train_damaged, train_cleaned,
                                   batch_size=comp_parameters.batch_size,
                                   epochs=comp_parameters.epochs,
                                   validation_split=0.3)

    # save the model in the output folder
    model_main.save(target_folder + globals.SAVE_MODEL_FILE)

    # try to predict cleaned images from the test dataset
    data.test_on_dataset(globals.TEST_DAMAGED, model_main, compute_time)

    # export results in the output folder (parameters, curves, and filters)
    vis.save_parameters(comp_parameters, target_folder)
    vis.plot_summary(model_history, target_folder)
    vis.plot_visualization_filter(model_main, target_folder)

    # display the computation time for this computation
    print(datetime.now().replace(microsecond=0) - start_time.replace(microsecond=0))
    vis.zip_email_files(target_folder)

    # delete the augmented images from directory
    if comp_parameters.augmentation:
        data.delete_augment_dataset()

    output.flush()

"""
    # parameters = ask_parameters()
    # cpu parameters = vis.Parameters(50, 10, globals.SGD, 0.01, globals.NO)
    # gpu parameters = vis.Parameters(2, 10, globals.SGD, 0.001, globals.NO, globals.VGG)
"""

parameters = vis.Parameters(64, 25, globals.ADAM, 0.001, globals.YES)
compile_whole(comp_parameters=parameters)


