# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!

# import pathloading
from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, QIntValidator, QFont
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
import qimage2ndarray
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel, QLineEdit,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy, QDesktopWidget, QGraphicsView, QGraphicsScene)
import Utility
import os
import numpy as np

# import SRGAN as gan
import tensorflow as tf
import time
# import skimage

class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.printer = QPrinter()
        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)
        self.setAcceptDrops(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.init_image_space()
        self.window_level = 40
        self.window_width = 400
        self.scene1 = QGraphicsScene()
        self.scene2 = QGraphicsScene()
        self.min_values = None
        self.max_values = None
        # self.net = gan.WGAN(nX = 512, nY = 512, channels = 1, nlayers = 3)
        
    def init_image_space(self):
        self.index = 0
        self.LD = None
        self.LD_screen = None
        self.HD = None
        self.HD_screen = None
        self.length_of_images = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(QDesktopWidget().availableGeometry().size() * 0.5)
        screen_width = QDesktopWidget().availableGeometry().size().width()
        subscreen = int((screen_width / 512.0) / 4.0) * 512
        self.scaleFactor = subscreen / 512
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(10, 10, subscreen+10, subscreen+10))
        self.graphicsView.setObjectName("graphicsView")

        self.graphicsView_2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setGeometry(QtCore.QRect(10+subscreen+10, 10, subscreen+10, subscreen+10))
        self.graphicsView_2.setObjectName("graphicsView_2")
        
        # self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        # self.textEdit.setGeometry(QtCore.QRect(100, 20+subscreen+50, 100, 40))
        
        # self.textEdit.setObjectName("textEdit")
        # self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        # self.textEdit_2.setGeometry(QtCore.QRect(205, 20+subscreen+50, 100, 40))
        # self.textEdit_2.setObjectName("textEdit_2")

        # self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        # self.pushButton.setGeometry(QtCore.QRect(310, 20+subscreen+50, 50, 40))
        # font = QtGui.QFont()
        # font.setFamily("Arial")
        # font.setPointSize(10)
        # font.setBold(True)
        # font.setWeight(75)
        # self.pushButton.setFont(font)
        # self.pushButton.setObjectName("pushButton")
        # self.pushButton.clicked.connect(self.update_windowing)

        MainWindow.setCentralWidget(self.centralwidget)

        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(10, 20+subscreen, subscreen*2+20, 31))
        
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalSlider.valueChanged.connect(self.changeValue)

        self.e1 = QLineEdit()
        self.e1.setValidator(QIntValidator())
        self.e1.setMaxLength(4)
        self.e1.setAlignment(Qt.AlignRight)
        self.e1.setFont(QFont("Arial",20))
        
        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1568, 31))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuImage = QtWidgets.QMenu(self.menubar)
        self.menuImage.setObjectName("menuImage")

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        # self.actionWindow_Level = QtWidgets.QAction(MainWindow)
        # self.actionWindow_Level.setObjectName("actionWindow_Level")
        # self.actionWindow_Level.triggered.connect(self.windowing_panel)
        
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionOpen.triggered.connect(self.open)
        
        self.actionOpenfolder = QtWidgets.QAction(MainWindow)
        self.actionOpenfolder.setObjectName("actionOpenFolders")
        self.actionOpenfolder.triggered.connect(self.openfolders)

        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionSave.triggered.connect(self.save)

        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionOpenfolder)
        self.menuFile.addAction(self.actionSave)
        # self.menuImage.addAction(self.actionWindow_Level)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuImage.menuAction())
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Low-dose CT image quality Improvement Project: LD2HD"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuImage.setTitle(_translate("MainWindow", "Image"))
        # self.actionWindow_Level.setText(_translate("MainWindow", "Window/Level.."))
        self.actionOpen.setText(_translate("MainWindow", "Open.."))
        self.actionOpenfolder.setText(_translate("MainWindow", "Open files.."))
        self.actionSave.setText(_translate("MainWindow", "Save.."))



    def changeValue(self, value):
        if value >= self.length_of_images:
            value = value -1
        self.show_images_LD_HD(value)

    def update_windowing(self):
        wl = int(self.textEdit.toPlainText())
        ww = int(self.textEdit_2.toPlainText())

        lowdose = np.copy(self.rawdata)
        fakehigh = np.copy(self.fake)

        lowdose = self.update_pixel_range(lowdose, wl, ww)
        fakehigh = self.update_pixel_range(fakehigh, wl, ww)

        self.LD_screen = self.ndarray_qimage_convert(lowdose)
        self.HD_screen = self.ndarray_qimage_convert(fakehigh)

    def read_dicom(self, files, path):
        return Utility.load_dicomseries(files, path)
            
    def update_pixel_range(self, image, level, width):
        return Utility.AdjustPixelRange(image, level, width)

    def ndarray_qimage_convert(self, image):
        images = []
        for i in range(image.shape[0]):
            images.append(qimage2ndarray.array2qimage(image[i]))
        return images


    def save(self):
        self.saving_Path = str(QFileDialog.getExistingDirectory(self, "Select Direct"))

        if not self.saving_Path == None:
            Utility.write_dicom(self.rootPath, self.saving_Path, self.files, self.fake)
        else:
            pass

    def open(self):
        self.init_image_space()
        self.rootPath = str(QFileDialog.getExistingDirectory(self, "Select Direct"))
        if self.rootPath == "":
            pass
        else:
            self.files = Utility.load_filenames(self.rootPath)
            self.files = Utility.extention_validation(self.files)
            self.files = Utility.ordering_check(self.rootPath, self.files)


            n_images = len(self.files)
            
            self.length_of_images = n_images
            self.horizontalSlider.setMaximum(n_images)
            
            self.LD = self.read_dicom(self.files, self.rootPath)
            self.rawdata = np.copy(self.LD)
            self.HD = np.zeros_like(self.LD, dtype=np.float32)

            self.LD = self.update_pixel_range(self.LD, self.window_level, self.window_width)
            self.LD_screen = self.ndarray_qimage_convert(self.LD)
            self.LD_screen = self.rescale_images_LD()
            self.show_images_LD(1)
            self.load_train_model()
   

    # def openfolders(self):
    #     # methods = ['MAP-NN', 'RED-CNN', 'SAGAN', 'WGAN', 'WGAN-GP-VGG']
    #     methods = ['MAP-NN']
    #     for method in methods:
    #         # path = "D:\LD2HD\\comparison\\Dicom\\{}\\L192".format(method)
    #         # files = Utility.load_filenames(path)
    #         # volume = self.read_dicom(files, path)
    #         volume = skimage.io.imread("D:\LD2HD\\comparison\\KAIST\\WaveResnet\\L192.tif")
            

    #         # saving_Path = path + "_fake"
    #         # print (saving_Path)
    #         # if not os.path.exists(saving_Path):
    #         #     os.mkdir(saving_Path)


    #         saving_Path = "D:\LD2HD\\comparison\\Dicom\\WaveResnet\\L192"

    #         path = "D:\LD2HD\\comparison\\Dicom\\origin\\L192"
    #         files = Utility.load_filenames(path)
    #         Utility.write_dicom(path, saving_Path, files, volume)


    def openfolders(self):

        nRandom = 2
        self.init_image_space()
        self.rootPath = str(QFileDialog.getExistingDirectory(self, "Select Direct"))
        if self.rootPath == "":
            pass
        else:
            folders = Utility.search_folders(self.rootPath)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.Session(config = config)
            session.run(tf.global_variables_initializer())
            self.net.restore_model(session, "model.ckpt-646")

        for folder in folders:
            path = os.path.join(self.rootPath, folder)
            starttime = time.time()

            files = Utility.load_filenames(path)
            files = Utility.extention_validation(files)
            files = Utility.ordering_check(path, files)

            rawdata = self.read_dicom(files, path)

            source = Utility.HU_intensity_norm(rawdata)
            source, min_values, max_values = Utility.ct_tanh_norm(rawdata)
            source = np.expand_dims(source, axis=3)
            temp = []

            volume = np.zeros((source.shape[0], source.shape[1], source.shape[2], nRandom), dtype='float32')

            for i in range(rawdata.shape[0]):
                start_time = time.time()
                index_start = i
                index_end = i+1
                LowDose = source[index_start:index_end]

                for iter in range (nRandom):
                    volume[i,:,:,iter] = self.net.predict(session, LowDose)[0,:,:,0]

                print(i, "--- %s seconds ---" % (time.time() - start_time))
            volume = volume.mean(axis=-1)
        
            for i in range(volume.shape[0]):
                volume[i] = (((volume[i] + 1.0)*0.5) * (max_values[i] - min_values[i])) + min_values[i]
        
            volume = np.clip(volume, -1024, 2048)
            volume = volume + 8.899902366042138

            saving_Path = path + "_fake"
            print (saving_Path)
            if not os.path.exists(saving_Path):
                os.mkdir(saving_Path)
        
            Utility.write_dicom(path, saving_Path, files, volume)
            print("--- %s seconds ---" % (time.time() - starttime))

        print ("Finished")
      
    def convert_folders(self, path):
        input("stop1")
        print (path)
        for dirpath, subdirs, fileList in os.walk(self.rootPath):
            print (dirpath)
            print (subdirs)
            print (fileList)
            input("stop2")

        self.files = Utility.load_filenames(self.rootPath)
        n_images = len(self.files)
        
        self.length_of_images = n_images
        self.horizontalSlider.setMaximum(n_images)
        
        self.LD = self.read_dicom(self.files, self.rootPath)
        print(self.LD.shape)
        self.rawdata = np.copy(self.LD)
        self.HD = np.zeros_like(self.LD, dtype=np.float32)

        self.LD = self.update_pixel_range(self.LD, self.window_level, self.window_width)
        self.LD_screen = self.ndarray_qimage_convert(self.LD)
        self.LD_screen = self.rescale_images_LD()
        self.show_images_LD(1)
        
        starttime = time.time()
        source = Utility.HU_intensity_norm(self.rawdata)
        source, self.min_values, self.max_values = Utility.ct_tanh_norm(self.rawdata)
        source = np.expand_dims(source, axis=3)

        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # session = tf.Session(config = config)

        # session.run(tf.global_variables_initializer())
   

    def rescale_images_LD(self):
        temp = []
        for index in range(self.length_of_images):
            pixelMap = QPixmap.fromImage(QImage(self.LD_screen[index]))
            temp.append(pixelMap.scaledToHeight(int(self.scaleFactor*512)))

        return temp

    def rescale_images_HD(self):
        temp = []
        for index in range(self.length_of_images):
            pixelMap = QPixmap.fromImage(QImage(self.HD_screen[index]))
            temp.append(pixelMap.scaledToHeight(int(self.scaleFactor*512)))

        return temp

    def show_images_LD(self, index):
        self.scene1.addPixmap(self.LD_screen[index])
        self.graphicsView.setScene(self.scene1)
        self.graphicsView.show()

    def show_images_HD(self, index):
        self.scene2.addPixmap(self.HD_screen[index])
        self.graphicsView_2.setScene(self.scene2)
        self.graphicsView_2.show()

    def show_images_LD_HD(self, index):
        self.scene1.addPixmap(self.LD_screen[index])
        self.scene2.addPixmap(self.HD_screen[index])
        self.graphicsView.setScene(self.scene1)
        self.graphicsView_2.setScene(self.scene2)        
        self.graphicsView.show()
        self.graphicsView_2.show()

    def load_train_model(self):
        starttime = time.time()
        source = Utility.HU_intensity_norm(self.rawdata)
        source, self.min_values, self.max_values = Utility.ct_tanh_norm(self.rawdata)
        source = np.expand_dims(source, axis=3)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config = config)

        session.run(tf.global_variables_initializer())
        self.net.restore_model(session, "model.ckpt-646")
        
        # model_path = "D:\\pyqt5\\viewerexample\\model.ckpt-646"
        # self.net.restore_model(session, model_path)
        # self.net.restore_model(session, os.path.join(os.getcwd(), "model", "model.ckpt-646"))
        
        nSlice = self.rawdata.shape[0]
        thickness = 2
        nIteration = int(nSlice / thickness)
        nResidual = nSlice - nIteration*thickness
        nRandom = 2
        
        volume = np.zeros((source.shape[0], source.shape[1], source.shape[2], nRandom), dtype='float32')

        for i in range(source.shape[0]):
            start_time = time.time()
            index_start = i
            index_end = i+1
            LowDose = source[index_start:index_end]

            for iter in range (nRandom):
                volume[i,:,:,iter] = self.net.predict(session, LowDose)[0,:,:,0]

            print(i, "--- %s seconds ---" % (time.time() - start_time))

        volume = volume.mean(axis=-1)
        
        for i in range(volume.shape[0]):
            volume[i] = (((volume[i] + 1.0)*0.5) * (self.max_values[i] - self.min_values[i])) + self.min_values[i]
        
        volume = np.clip(volume, -1024, 2048)
        volume = volume + 8.899902366042138

        # for i in range(nIteration):
        #     start_time = time.time()
        #     index_start = i*thickness
        #     index_end   = (i+1)*thickness
        #     LowDose = source[index_start:index_end]
            
        #     temp = []
        #     for j in range(nRandom):
        #         temp.append(self.net.predict(session, LowDose))

        #     Fake_HighDose = np.asarray(temp)
        #     Fake_HighDose = np.mean(Fake_HighDose, axis=0)
        #     Fake_HighDose = Utility.reverse_ct_tanh_norm(Fake_HighDose, self.min_values[index_start:index_end], self.max_values[index_start:index_end])
        #     Fake_HighDose = Fake_HighDose + 8.899902366042138
            
        #     self.HD[index_start:index_end] = np.squeeze(Fake_HighDose)
        #     print(i, "--- %s seconds ---" % (time.time() - start_time)) 

        # start_time = time.time()
        # LowDose = source[-nResidual:]
        # temp = []
        # for j in range(nRandom):
        #     temp.append(self.net.predict(session, LowDose))

        # Fake_HighDose = np.asarray(temp)
        # Fake_HighDose = np.mean(Fake_HighDose, axis=0)
        # Fake_HighDose = Utility.reverse_ct_tanh_norm(Fake_HighDose, self.min_values[-nResidual:], self.max_values[-nResidual:])
        # Fake_HighDose = Fake_HighDose + 8.899902366042138
        
        # self.HD[-nResidual:] = np.squeeze(Fake_HighDose)
        # print(i+1, "--- %s seconds ---" % (time.time() - start_time))

        print("Finished %s seconds ---" % (time.time() - starttime))

        self.HD = volume
        self.fake = np.copy(self.HD)

        self.HD = self.update_pixel_range(self.HD, self.window_level, self.window_width)
        self.HD_screen = self.ndarray_qimage_convert(self.HD)
        self.HD_screen = self.rescale_images_HD()
        self.show_images_HD(1)

    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel.pixmap())

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def about(self):
        QMessageBox.about(self, "About Image Viewer",
                "<p>The <b>Image Viewer</b> example shows how to combine "
                "QLabel and QScrollArea to display an image. QLabel is "
                "typically used for displaying text, but it can also display "
                "an image. QScrollArea provides a scrolling view around "
                "another widget. If the child widget exceeds the size of the "
                "frame, QScrollArea automatically provides scroll bars.</p>"
                "<p>The example demonstrates how QLabel's ability to scale "
                "its contents (QLabel.scaledContents), and QScrollArea's "
                "ability to automatically resize its contents "
                "(QScrollArea.widgetResizable), can be used to implement "
                "zooming and scaling features.</p>"
                "<p>In addition the example shows how to use QPainter to "
                "print an image.</p>")

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                                + ((factor - 1) * scrollBar.pageStep()/2)))


if __name__ == "__main__":
    import sys
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
