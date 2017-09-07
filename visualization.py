import csv
import math
import os
import shutil
import smtplib
import zipfile
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import globals


class Parameters:
    """ Class created to store all the parameters needed at each computation in one variable.
    """
    # have all parameters in one variable (one class)
    def __init__(self, save_batch, save_epochs, save_opt, save_lr, save_augm, save_model='simple'):
        self.model = save_model
        self.batch_size = save_batch
        self.epochs = save_epochs
        self.optimizer = save_opt
        self.learning_rate = save_lr
        self.augmentation = save_augm


def create_folder():
    """ Creates output folder.

        Creates output folder according to current time.

        # Arguments
            None
        # Returns
            creation time and created folder
    """
    # save the model in h5 file
    end_time = datetime.now().strftime("%d-%m_%Hh%M")
    created_folder = os.path.join(globals.DIR, globals.OUTPUTS, end_time)
    os.makedirs(created_folder)
    return end_time, created_folder


def save_parameters(save_par, save_folder):
    """ Saves all parameters in a csv file, inside the output folder

        Gets all parameters from the object, and saves them in the output/time/parameters.csv file.

        # Arguments
            save_par: class containing all the parameters
            save_folder: destination folder
        # Returns
            None
    """
    # export all parameters to csv using dictionary
    with open(save_folder + '/parameters.csv', 'w', newline='') as parameters:
        writer = csv.writer(parameters)
        for key, value in save_par.__dict__.items():
            writer.writerow([key, value])


def plot_summary(history, save_folder):
    """ Plots summary of training (accuracy and loss)

        Gets loss and accuracy from the history created from model training.
        Subplots them and saves them in a file.

        # Arguments
            history: history data obtained from training
            save_folder: destination folder for the file
        # Returns
            None
    """
    # plot loss on one subplot
    plt.figure()
    plt.rcParams["figure.figsize"] = [15, 10]
    plt.subplot(121)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    # and accuracy functions on the second one
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    # save the file
    plt.savefig(save_folder + '/summary.png')
    plt.close()


def plot_filters(model_plot, layer_index, save_folder):
    """ Plots the filter of a given layer in a given model.

        Obtains layers matrices from the model. Computes rows and cols for display
        purposes, then nicely displays all filters in a matshow. Saves the matrix
        in a file in the given folder.

        # Arguments
            model_plot: reference model to get layers from
            layer_index: index of reference layer
            save_folder: destination folder
        # Returns
            None
    """
    # get the filters matrix and format to plot in mat
    filters = model_plot.layers[layer_index].get_weights()[0]
    layer_name = model_plot.layers[layer_index].get_config().get('name')
    filters = filters.reshape(-1, filters.shape[-3], filters.shape[-4])
    filters_number = filters.shape[0]
    rows = 2 * math.sqrt(filters_number / 4 * (1 + (math.log2(filters_number / 4)) % 2))
    cols = filters_number / rows
    fig = plt.figure(figsize=(17.50, 8.75))

    # add each filter to the matrix
    for j in range(filters_number):
        ax = fig.add_subplot(cols, rows, j+1)
        ax.matshow(filters[j], cmap=cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()

    # save the file
    plt.savefig(save_folder+'/'+str(layer_name)+'.png')


def plot_visualization_filter(model_vis, plot_folder):
    """ Plots filters of each layer in the model in the given folder.

        Obtains all convolution layers from the folder, get the filters and plots them.

        # Arguments
            model_vis: reference model to get layers from
            plot_folder: destination folder
        # Returns
            None
    """
    # plot filter matrix
    conv_layers = []
    for layer_index in range(len(model_vis.get_config())):
        layer_config = model_vis.layers[layer_index].get_config()
        if 'filters' in layer_config:
            conv_layers.append(layer_index)
    # only plot the first layer (too many filters for deeper layers)
    plot_filters(model_vis, conv_layers[0], plot_folder)


def zip_email_files(folder_to_zip):
    """ Zips all the output files and emails them to address specified in the globals file.

        Finds all the files in the output folder, zips them and puts the compressed
        file in the same folder.
        Then, depending on login information (in email.txt), send email containing zip
        file as attachment to specified email address.

        # Arguments
            folder_to_zip: reference folder to zip and email
        # Returns
            None
    """
    file_zip_name = os.path.basename(folder_to_zip)
    zip_path = folder_to_zip + '/' + file_zip_name
    # create the zip file
    zip_file = zipfile.ZipFile(folder_to_zip + '.zip', 'w')
    folder_size = len(folder_to_zip) + 1
    # zip all the files but the h5 saved file (can be too heavy for the email, even zipped
    for base, dirs, files in os.walk(folder_to_zip):
        for file in files:
            fn = os.path.join(base, file)
            if not fn.endswith('h5'):
                zip_file.write(fn, fn[folder_size:])
    zip_file.close()
    # move the zip file to inside the folder
    shutil.move(folder_to_zip+'.zip', zip_path+'.zip')

    # get login credentials from text file (security)
    with open('email.txt') as email_login:
        login = email_login.read().splitlines()
        msg = MIMEMultipart()
        address_from = login[1]
        address_to = globals.EMAIL_ADDRESS

        # specify email headers
        msg['From'] = address_from
        msg['To'] = address_to
        msg['Subject'] = 'Outputs: ' + file_zip_name

        body = "Find attached the zip output"
        msg.attach(MIMEText(body, 'plain'))

        # attach the zip file
        filename = file_zip_name + '.zip'
        attachment = open(zip_path + '.zip', "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
        msg.attach(part)

        # send the email
        server = smtplib.SMTP(login[0], 587)
        server.starttls()
        server.login(address_from, login[2])
        text = msg.as_string()
        server.sendmail(address_from, address_to, text)
        server.quit()
