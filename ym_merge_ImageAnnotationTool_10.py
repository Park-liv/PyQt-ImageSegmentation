import sys
import os 
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QWidget, QHBoxLayout,\
    QVBoxLayout, QAction, QFileDialog, QGraphicsView, QGraphicsScene, QCheckBox, QComboBox, QPushButton,\
         QInputDialog, qApp, QLineEdit, QMessageBox, QRadioButton, QGroupBox, QSlider)
from PyQt5.QtGui import QPixmap, QIcon, QImage, QWheelEvent, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from PyQt5 import QtWidgets, QtCore
import natsort
import numpy as np
import SimpleITK as itk
import qimage2ndarray
import math
import copy
import voxel
import cv2

# import import_ipynb
import ym_model_1
import tensorflow as tf
from tensorflow import keras

class MyWidget(QWidget): 
    def __init__(self): 
        super().__init__() 

        self.scene_1 = QGraphicsScene()
        self.scene_2 = QGraphicsScene()
        self.view_1 = QGraphicsView(self.scene_1) 
        self.view_2 = QGraphicsView(self.scene_2)

        self.deleteCurMaskBtn = QPushButton('Delete Mask(Not exist)', self)
        self.addMaskBtn = QPushButton('&Add Mask', self)
        self.maskComboBox = QComboBox(self)
        self.showAllMaskCheckBox = QCheckBox('Show all masks', self)
        self.maskCheckBox = QCheckBox('Masking', self)
        self.blendCheckBox = QCheckBox('&Blended Mask on', self)
        self.blendCheckBox.setShortcut('X')
        self.penSizeEdit = QLineEdit(self)
        self.penSizeEdit.setFixedWidth(30)
        self.transparentSlider = QSlider(Qt.Horizontal)
        self.transparentSlider.setFixedWidth(300)

        self.dialogBtn = QPushButton('&ImgNum', self)  
        self.previousBtn = QPushButton('&previous', self)
        self.nextBtn = QPushButton('&next', self)
        self.morphBtn = QPushButton('&morph', self)
        self.predictionCheckBox = QCheckBox('&Prediction', self)

        self.lbl_pen_size = QLabel('Pen & Eraser size', self)
        self.lbl_pen_size.setAlignment(Qt.AlignCenter)
        self.lbl_pos = QLabel()
        self.lbl_pos.setAlignment(Qt.AlignCenter)
        self.lbl_image_fname = QLabel('Image file is Not Opened.', self)
        self.lbl_image_fname.setAlignment(Qt.AlignCenter)
        self.lbl_mask_fname = QLabel('Mask file is Not Opened.', self)
        self.lbl_mask_fname.setAlignment(Qt.AlignCenter)

        self.changeAxisBtn1 = QRadioButton('Axial', self)
        self.changeAxisBtn2 = QRadioButton('Coronal', self)
        self.changeAxisBtn3 = QRadioButton('Sagittal', self)
        self.changeAxisBtn1.setChecked(True)
        
        self.gAxisbox = QGroupBox('Axis view')
        self.hAxisbox = QHBoxLayout()
        self.hAxisbox.addWidget(self.changeAxisBtn1)
        self.hAxisbox.addWidget(self.changeAxisBtn2)
        self.hAxisbox.addWidget(self.changeAxisBtn3)
        self.gAxisbox.setLayout(self.hAxisbox)
        self.gAxisbox.setEnabled(False)
        
        self.hViewbox = QHBoxLayout()
        self.hViewbox.addWidget(self.view_1)
        self.hViewbox.addWidget(self.view_2)

        self.view_1.wheelEvent = self.wheelEvent
        self.view_2.wheelEvent = self.wheelEvent

        self.hOptionbox_1 = QHBoxLayout()
        self.hOptionbox_1.addWidget(self.deleteCurMaskBtn)
        self.hOptionbox_1.addWidget(self.addMaskBtn)
        self.hOptionbox_1.addWidget(self.maskComboBox)
        self.hOptionbox_1.addWidget(self.lbl_pos)
        self.hOptionbox_1.addWidget(self.morphBtn)
        self.hOptionbox_1.addWidget(self.previousBtn)
        self.hOptionbox_1.addWidget(self.nextBtn)
        self.hOptionbox_1.addWidget(self.dialogBtn)

        self.vLabelBox = QVBoxLayout()
        self.vLabelBox.addWidget(self.lbl_image_fname)
        self.vLabelBox.addWidget(self.lbl_mask_fname)

        self.hOptionbox_2 = QHBoxLayout()
        self.hOptionbox_2.addLayout(self.vLabelBox)
        self.hOptionbox_2.addStretch(1)
        self.hOptionbox_2.addWidget(self.gAxisbox)
        self.hOptionbox_2.addWidget(self.lbl_pen_size)
        self.hOptionbox_2.addWidget(self.penSizeEdit)
        self.hOptionbox_2.addWidget(self.predictionCheckBox)
        self.hOptionbox_2.addWidget(self.showAllMaskCheckBox)
        self.hOptionbox_2.addWidget(self.maskCheckBox)
        self.hOptionbox_2.addWidget(self.blendCheckBox)
        self.hOptionbox_2.addWidget(self.transparentSlider)
        
        self.vbox = QVBoxLayout()
        self.vbox.addLayout(self.hViewbox)
        self.vbox.addLayout(self.hOptionbox_1)
        self.vbox.addLayout(self.hOptionbox_2)
        
        self.setLayout(self.vbox)

    def resizeEvent(self, event):
        if event.oldSize().width() < 1 or event.oldSize().height() < 1:
            return
        self.view_1.scale(event.size().width()/event.oldSize().width(), \
            event.size().height()/event.oldSize().height())
        self.view_2.scale(event.size().width()/event.oldSize().width(), \
            event.size().height()/event.oldSize().height())

class PredictionWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.cur_idx = 0
        self.cur_image = None
        self.pred_volume = None
        self.transparent = 0
    
    def initUI(self):
        self.scene_pred = QGraphicsScene()
        self.view_pred = QGraphicsView(self.scene_pred)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.view_pred)
        self.setLayout(self.layout)
        self.setWindowTitle('Prediction Image')

    def refresh(self):
        x, y = self.pred_volume.shape[1], self.pred_volume.shape[2]
        r_img_arr = np.array([[[255, 0, 0, math.floor(255*self.transparent)]] * y] * x)
        g_img_arr = np.array([[[0, 255, 0, math.floor(255*self.transparent)]] * y] * x)
        b_img_arr = np.array([[[0, 0, 255, math.floor(255*self.transparent)]] * y] * x)

        r_label = np.where(self.pred_volume[self.cur_idx] == 1, 1, 0).copy().reshape(x, y, 1)
        g_label = np.where(self.pred_volume[self.cur_idx] == 2, 1, 0).copy().reshape(x, y, 1)
        b_label = np.where(self.pred_volume[self.cur_idx] == 3, 1, 0).copy().reshape(x, y, 1)

        new_img = np.multiply(r_img_arr, r_label) + np.multiply(g_img_arr, g_label) + np.multiply(b_img_arr, b_label)
        self.scene_pred.clear()
        self.scene_pred.addPixmap(QPixmap.fromImage(QImage(self.cur_image)))
        self.scene_pred.addPixmap(QPixmap.fromImage(QImage(qimage2ndarray.array2qimage(new_img))))

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # self.window_level = 40
        # self.window_width = 400
        self.window_level = 220
        self.window_width = 740
        self.deltaWL = 0
        self.deltaWW = 0

        self.Nx = 0 
        self.Ny = 0 
        self.NofI = 0 

        self.cur_idx = 0 
        self.cur_image = [] 
        self.EntireImage = [] 
        self.adjustedImage = []
        
        self.is_opened = False
        self.is_predicted = False
        self.Lclicked = False
        self.Rclicked = False
        self.lastPoint = QPoint()
        self.mask_arrList = []
        self.drawn_imgList = []
        self.onCtrl = False
        self.onShift = False
        self.pen_size = 10
        self.py_raw = voxel.PyVoxel()
        self.cur_axis = 0
        self.image_fname = ''
        self.mask_fname = ''
        self.transparent = 0.5
        self.color_list = [Qt.red, Qt.green, Qt.blue]

        self.wg = MyWidget()
        self.pred_win = PredictionWindow()
        self.setCentralWidget(self.wg)
        self.initUI()
        
    def initUI(self):
        openRaw = QAction('Open Raw File', self)
        openRaw.triggered.connect(self.openImageRaw)
        openIMA = QAction('Open IMA File', self)
        openIMA.triggered.connect(self.openImageIMA)
        exitAction = QAction('Quit', self)
        exitAction.triggered.connect(qApp.quit)
        saveNpyAction = QAction('Save Masks As Npy', self)
        saveNpyAction.triggered.connect(self.saveMasksAsNpy)
        saveBinAction = QAction('Save Masks As Bin', self)
        saveBinAction.triggered.connect(self.saveMasksAsBin)
        loadNpyAction = QAction('Load Masks From Npy', self)
        loadNpyAction.triggered.connect(self.loadMasksNpy)
        loadBinAction = QAction('Load Masks From Bin', self)
        loadBinAction.triggered.connect(self.loadBinMasks)
        adjustAction = QAction('Adjust', self)
        adjustAction.triggered.connect(self.adjustImage)

        self.statusBar()

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        filemenu = menubar.addMenu('&File')
        filemenu.addAction(openRaw)
        filemenu.addAction(openIMA)
        filemenu.addAction(saveNpyAction)
        filemenu.addAction(saveBinAction)
        filemenu.addAction(loadNpyAction)
        filemenu.addAction(loadBinAction)
        filemenu.addAction(exitAction)
        imagemenu = menubar.addMenu('&Image')
        imagemenu.addAction(adjustAction)
        
        self.wg.deleteCurMaskBtn.clicked.connect(self.deleteMask)
        self.wg.addMaskBtn.clicked.connect(self.addMask)

        self.wg.maskCheckBox.stateChanged.connect(self.onMasking)
        self.wg.maskComboBox.activated.connect(self.maskComboBoxActivated)
        self.wg.maskComboBox.setMaxCount(len(self.color_list))
        self.wg.blendCheckBox.stateChanged.connect(self.onBlendedMask)
        self.wg.showAllMaskCheckBox.stateChanged.connect(self.showAllMasks)
        self.wg.morphBtn.clicked.connect(self.morphBtn_clicked)
        self.wg.predictionCheckBox.stateChanged.connect(self.predict_currentImg)

        self.wg.transparentSlider.setRange(1, 10)
        self.wg.transparentSlider.setSingleStep(1)
        self.wg.transparentSlider.setValue(self.transparent*10)
        self.wg.transparentSlider.valueChanged.connect(self.setTransparentValue)

        self.wg.penSizeEdit.textChanged[str].connect(self.setPenSize)
        self.wg.penSizeEdit.setText(str(self.pen_size))

        self.wg.dialogBtn.clicked.connect(self.showDialog)
        self.wg.previousBtn.clicked.connect(self.previousBtn_clicked)
        self.wg.nextBtn.clicked.connect(self.nextBtn_clicked)

        self.wg.changeAxisBtn1.clicked.connect(self.changeAxis)
        self.wg.changeAxisBtn2.clicked.connect(self.changeAxis)
        self.wg.changeAxisBtn3.clicked.connect(self.changeAxis)

        self.wg.view_1.setMouseTracking(True)
        self.wg.view_2.setMouseTracking(True)

        self.wg.scene_1.mouseMoveEvent = self.mouseMoveEvent
        self.wg.scene_1.mousePressEvent = self.mousePressEvent
        self.wg.scene_1.mouseReleaseEvent = self.mouseReleaseEvent
        self.wg.scene_2.mouseMoveEvent = self.mouseMoveEvent
        self.wg.scene_2.mousePressEvent = self.mousePressEvent
        self.wg.scene_2.mouseReleaseEvent = self.mouseReleaseEvent

        self.setWindowTitle('Image Labeling')
        self.setGeometry(300, 300, 1100, 600)
        self.show()
    
    def openImageRaw(self):
        try:
            # Rad for .raw file
            image_path = QFileDialog.getOpenFileName(self, "Select File", './')[0]
            self.image_fname = image_path.split('/')[-1]
            self.wg.lbl_image_fname.setText('Image file name : ' + self.image_fname)
            self.py_raw.ReadFromRaw(image_path)
            ImgArray = self.py_raw.m_Voxel

            self.EntireImage = np.asarray(ImgArray, dtype=np.float32) 
            self.EntireImage = np.squeeze(self.EntireImage)
            self.NofI = self.EntireImage.shape[0]  
            self.Nx = self.EntireImage.shape[1] 
            self.Ny = self.EntireImage.shape[2] 
            self.mask_arrList = np.zeros((1, self.NofI, self.Nx, self.Ny))
            self.refresh()
            self.is_opened = True
            self.is_predicted = False
            self.wg.gAxisbox.setEnabled(True)
            if not self.wg.changeAxisBtn1.isChecked():
                self.wg.changeAxisBtn1.toggle()
        except:
            print('openImageRaw Error')

    def openImageIMA(self):
        try:
            # Read for Dicom series files
            image_path = str(QFileDialog.getExistingDirectory(self, "Select Directory", './'))
            self.image_fname = image_path.split('/')[-1]
            self.wg.lbl_image_fname.setText('Image file name : ' + self.image_fname)
            reader = itk.ImageSeriesReader() 
            dicom_names = reader.GetGDCMSeriesFileNames(image_path)
            dicom_names = natsort.natsorted(dicom_names)
            reader.SetFileNames(dicom_names)
            images = reader.Execute()
            ImgArray = itk.GetArrayFromImage(images)

            self.EntireImage = np.asarray(ImgArray, dtype=np.float32) 
            self.EntireImage = np.squeeze(self.EntireImage)
            self.NofI = self.EntireImage.shape[0]
            self.Nx = self.EntireImage.shape[1]
            self.Ny = self.EntireImage.shape[2]
            self.mask_arrList = np.zeros((1, self.NofI, self.Nx, self.Ny))
            self.refresh()
            self.is_opened = True
            self.is_predicted = False
            self.wg.gAxisbox.setEnabled(True)
            if not self.wg.changeAxisBtn1.isChecked():
                self.wg.changeAxisBtn1.toggle()
        except:
            print('openImageIMA Error')

    def adjustImage(self):
        level, ok = QInputDialog.getInt(self, 'Level', 'Level Set', value=self.window_level)
        width, ok = QInputDialog.getInt(self, 'Width', 'Width Set', value=self.window_width)
        self.window_level = level
        self.window_width = width
        self.refresh()

    def showDialog(self):
        num, ok = QInputDialog.getInt(self, 'Input ImageNumber', 'Enter Num', value=self.cur_idx+1)
        self.cur_idx = num - 1
        if self.cur_idx > self.NofI-1:
            self.cur_idx = self.NofI-1
        elif self.cur_idx < 0:
            self.cur_idx = self.NofI-224
        self.refresh()

    def refresh(self): 
        try:
            cur_mask_index = self.wg.maskComboBox.currentIndex()
            self.wg.maskComboBox.clear()
            for i in range(self.mask_arrList.shape[0]):
                self.wg.maskComboBox.addItem('Mask' + str(i + 1))
            if cur_mask_index >= 0: self.wg.maskComboBox.setCurrentIndex(cur_mask_index)

            self.cur_orginal_image = self.EntireImage[self.cur_idx]
            self.cur_img_arr = self.AdjustPixelRange(self.cur_orginal_image, self.window_level, self.window_width)
            self.cur_image = qimage2ndarray.array2qimage(self.cur_img_arr)
            cur_image = QPixmap.fromImage(QImage(self.cur_image))

            self.wg.scene_1.clear()
            self.wg.scene_2.clear()
            self.wg.scene_1.addPixmap(cur_image)
            self.wg.scene_2.addPixmap(cur_image)
            self.wg.view_1.setScene(self.wg.scene_1)
            self.wg.view_2.setScene(self.wg.scene_2)

            mask = self.label2image(self.mask_arrList[self.wg.maskComboBox.currentIndex(), self.cur_idx])
            self.cur_maskPixmap = QPixmap.fromImage(QImage(mask))
            self.drawn_imgList = [mask]
            self.wg.scene_2.addPixmap(self.cur_maskPixmap)
            
            self.wg.deleteCurMaskBtn.setText('Delete Mask {}'.format(self.wg.maskComboBox.currentIndex()+1))
            if self.wg.maskCheckBox.isChecked(): self.wg.maskCheckBox.toggle()
            if self.wg.blendCheckBox.isChecked(): self.wg.blendCheckBox.toggle()
            if self.wg.showAllMaskCheckBox.isChecked(): self.showAllMasks(Qt.Checked)
            if self.wg.predictionCheckBox.isChecked():
                if self.pred_win.isVisible(): self.predict_currentImg(Qt.Checked)
                else: self.wg.predictionCheckBox.toggle()
        except:
            print('refresh Error')
        
    def previousBtn_clicked(self):
        try:
            if self.is_opened:
                self.cur_idx = self.cur_idx - 1
                if self.cur_idx < 0: 
                    self.cur_idx = 0
                self.refresh()
        except:
            print('previousBtn_clicked Error')

    def nextBtn_clicked(self):
        try:
            if self.is_opened:
                self.cur_idx = self.cur_idx + 1
                if self.cur_idx > self.NofI-1:
                    self.cur_idx = self.NofI-1
                self.refresh()
        except:
            print('nextBtn_clicked Error')

    def AdjustPixelRange(self, image, level, width):
        Lower = level - (width/2.0)
        Upper = level + (width/2.0)
        range_ratio = (Upper - Lower) / 256.0
        img_adjusted = (image - Lower)/range_ratio
        image = img_adjusted.clip(0, 255)
        return image

    def wheelEvent(self, event):
        try:
            if self.is_opened:
                n_scroll = int(event.angleDelta().y() / 120)
                
                self.cur_idx = self.cur_idx + n_scroll
                if self.cur_idx < 0:
                    self.cur_idx = 0
                if self.cur_idx > self.NofI-1:
                    self.cur_idx = self.NofI-1
                self.refresh() 
        except:
            print('wheelEvent Error')

    def mouseMoveEvent(self, event):
        try:
            if self.is_opened:
                if self.Lclicked and self.Rclicked:
                    rX = self.lastPoint.x()
                    rY = self.lastPoint.y()
                    
                    mX = event.scenePos().x()
                    mY = event.scenePos().y()

                    square = (rX - mX)*(rX - mX) + (rY - mY)*(rY - mY)
                    dist = math.sqrt(square) / 5

                    if rX < mX: self.deltaWL  = dist                
                    else: self.deltaWL  = -dist
                    if rY < mY: self.deltaWW = -dist
                    else: self.deltaWW = dist
                    self.window_level = self.window_level + self.deltaWL
                    self.window_width = self.window_width + self.deltaWW

                    if self.window_width <= 0: self.window_width = 0
                    elif self.window_width > 900: self.window_width = 900

                    if self.window_level < -250: self.window_level = -250
                    elif self.window_level > 100: self.window_level = 100
                    self.refresh()

                if self.Lclicked:
                    painter = QPainter(self.cur_maskPixmap)
                    painter.setPen(QPen(self.color_list[self.wg.maskComboBox.currentIndex()], self.pen_size, Qt.SolidLine))
                    if self.onCtrl:
                        painter.drawLine(self.lastPoint, event.scenePos().toPoint())
                    elif self.onShift:
                        r = QRect(self.lastPoint, self.pen_size * QSize())
                        r.moveCenter(event.scenePos().toPoint())
                        painter.setCompositionMode(QPainter.CompositionMode_Clear)
                        painter.eraseRect(r)
                    self.wg.scene_2.removeItem(self.wg.scene_2.items()[0])
                    self.wg.scene_2.addPixmap(self.cur_maskPixmap)
                
                self.lastPoint = event.scenePos().toPoint()

                if (self.lastPoint.x() >= 0) and (self.lastPoint.x() < self.Nx):
                    if (self.lastPoint.y() >= 0) and (self.lastPoint.y() < self.Ny):
                        value = self.cur_orginal_image[self.lastPoint.x(), self.lastPoint.y()]
                    else: value = -1
                else: value = -1

                txt = "x={0}, y={1}, z={2}, image value={3}".format(\
                    self.lastPoint.x(), self.lastPoint.y(), self.cur_idx+1, value) 
                self.wg.lbl_pos.setText(txt)
        except:
            print('mouseMoveEvent Error')

    def mousePressEvent(self, event):
        try:
            if event.button() == Qt.LeftButton:
                self.Lclicked = True
            if event.button() == Qt.RightButton:
                self.Rclicked = True
        except:
            print('mousePressEvent Error')

    def mouseReleaseEvent(self, event):
        try:
            if event.button() == Qt.LeftButton:
                if self.Lclicked:
                    self.mask_arrList[self.wg.maskComboBox.currentIndex(), self.cur_idx] = \
                        self.image2label(self.cur_maskPixmap.toImage())
                    self.drawn_imgList.append(self.cur_maskPixmap.toImage())
                    self.refreshMaskView()
                self.Lclicked = False
            if event.button() == Qt.RightButton:
                self.Rclicked = False
        except:
            print('mouseReleaseEvent Error')

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.onCtrl = True
        if event.key() == Qt.Key_Shift:
            self.onShift = True
        if self.onCtrl and event.key() == Qt.Key_Z:
            self.erasePreviousLine()
        if self.onCtrl and event.key() == Qt.Key_Plus:
            self.wg.view_2.scale(1.25, 1.25)
        if self.onCtrl and event.key() == Qt.Key_Minus:
            self.wg.view_2.scale(0.8, 0.8)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.onCtrl = False
        if event.key() == Qt.Key_Shift:
            self.onShift = False
        
    def erasePreviousLine(self):
        if len(self.drawn_imgList) > 1:
            del self.drawn_imgList[len(self.drawn_imgList)-1]
            self.cur_maskPixmap = QPixmap.fromImage(QImage(self.drawn_imgList[len(self.drawn_imgList)-1]))
            self.mask_arrList[self.wg.maskComboBox.currentIndex(), self.cur_idx] = \
                self.image2label(self.cur_maskPixmap.toImage())
            self.refreshMaskView()

    def onMasking(self, state):
        try:
            if Qt.Checked == state:
                origin_arr = np.array(qimage2ndarray.rgb_view(self.cur_image))
                mask_arr = self.mask_arrList[self.wg.maskComboBox.currentIndex(), self.cur_idx].copy()
                mask_arr = np.expand_dims(mask_arr, axis=2)
                
                self.masked_arr = np.multiply(origin_arr, mask_arr)
                self.masked_qimg = qimage2ndarray.array2qimage(self.masked_arr)
                self.masked_pixmap = QPixmap.fromImage(QImage(self.masked_qimg))

                self.wg.scene_2.addPixmap(self.masked_pixmap)
            else:
                self.wg.scene_2.removeItem(self.wg.scene_2.items()[0])
        except:
            print('onMasking Error')
                
    def onBlendedMask(self, state):
        try:
            if Qt.Checked == state:
                masked_qimg = self.label2image(self.mask_arrList[self.wg.maskComboBox.currentIndex(), self.cur_idx])
                masked_arr = self.bgra2rgba(qimage2ndarray.byte_view(masked_qimg))
                masked_alpha_arr = masked_arr[:, :, 3].copy()
                masked_arr[:, :, 3] = masked_alpha_arr * self.transparent

                blended_mask = qimage2ndarray.array2qimage(masked_arr)
                blended_mask = QPixmap.fromImage(QImage(blended_mask))

                self.wg.scene_2.removeItem(self.wg.scene_2.items()[0])
                self.wg.scene_2.addPixmap(blended_mask)
            else:
                self.wg.scene_2.removeItem(self.wg.scene_2.items()[0])
                self.wg.scene_2.addPixmap(self.cur_maskPixmap)
        except:
            print('onBlendedMask Error')

    def addMask(self):
        try:
            if self.mask_arrList.shape[0] < len(self.color_list):
                self.mask_arrList = np.concatenate((self.mask_arrList, np.zeros((1, self.NofI, self.Nx, self.Ny))), axis=0)
                self.wg.maskComboBox.addItem('Mask' + str(self.mask_arrList.shape[0]))
                self.maskComboBoxActivated(self.mask_arrList.shape[0]-1)
                self.wg.maskComboBox.setCurrentIndex(self.mask_arrList.shape[0]-1)
            else:
                QMessageBox.warning(self, "Warining" ,"Maximum mask count is 3")
        except:
            print('addMask Error')

    def deleteMask(self): 
        try:
            if self.mask_arrList.shape[0] > 1:
                self.mask_arrList = np.delete(self.mask_arrList, self.wg.maskComboBox.currentIndex(), axis=0)
                print(self.mask_arrList.shape)
                self.wg.maskComboBox.removeItem(self.wg.maskComboBox.currentIndex())
                cur_mask_index = self.wg.maskComboBox.currentIndex()
                self.wg.maskComboBox.clear()
                for i in range(self.mask_arrList.shape[0]):
                    self.wg.maskComboBox.addItem('Mask' + str(i + 1))
                self.maskComboBoxActivated(cur_mask_index)
                self.wg.maskComboBox.setCurrentIndex(cur_mask_index)
            else:
                return
        except:
            print('deleteMask Error')

    def maskComboBoxActivated(self, index):
        mask = self.label2image(self.mask_arrList[index, self.cur_idx])
        self.cur_maskPixmap = QPixmap.fromImage(QImage(mask))
        self.drawn_imgList = [mask]
        self.wg.deleteCurMaskBtn.setText('Delete Mask {}'.format(index+1))
        self.refreshMaskView()

    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    def bgra2rgba(self, bgra):
        rgba = bgra.copy()
        rgba[:, :, 0], rgba[:, :, 2] = bgra[:, :, 2], bgra[:, :, 0]
        
        return rgba

    def image2label(self, image):
        alpha_arr = qimage2ndarray.alpha_view(image)
        return np.where(alpha_arr > 0, 1, 0)

    def label2image(self, label):
        x, y = label.shape[0], label.shape[1]
        color = [0, 0, 0, 255]
        color[self.wg.maskComboBox.currentIndex()] = 255

        img_arr = np.array([[color] * y] * x)
        new_label = label.copy().reshape(x, y, 1)
        return qimage2ndarray.array2qimage(np.multiply(img_arr, new_label))

    def refreshMaskView(self):
        self.wg.scene_2.clear()
        self.wg.scene_2.addPixmap(QPixmap.fromImage(QImage(self.cur_image)))
        self.wg.scene_2.addPixmap(self.cur_maskPixmap)
        if self.wg.maskCheckBox.isChecked(): self.wg.maskCheckBox.toggle()
        if self.wg.blendCheckBox.isChecked(): self.wg.blendCheckBox.toggle()

    def setPenSize(self, text):
        try:
            self.pen_size = int(text)
        except:
            return
    
    def setTransparentValue(self, value):
            self.transparent = value/10
            print(self.transparent)
            if self.wg.blendCheckBox.isChecked(): self.onBlendedMask(Qt.Checked)

    def saveMasksAsNpy(self):
        try:
            if not self.wg.changeAxisBtn1.isChecked():
                self.wg.changeAxisBtn1.toggle()
                self.changeAxis()

            save_fname = QFileDialog.getSaveFileName(self, "Save Masks as npy", './untitled.npy')[0]
            if len(save_fname) < 1: 
                return
            if '.npy' not in save_fname:
                save_fname = save_fname + '.npy'

            for i in range(self.mask_arrList.shape[0]):
                # print(save_fname[:-4] + '_{}'.format(i+1) + save_fname[-4:])
                np.save(save_fname[:-4] + '_{}'.format(i+1) + save_fname[-4:], self.mask_arrList[i])

            QMessageBox.information(self, 'Save All Masks', "All masks is saved.", \
                    QMessageBox.Ok, QMessageBox.Ok)
        except:
            print('saveMasksAsNpy Error')

    def saveMasksAsBin(self):
        try:
            if not self.wg.changeAxisBtn1.isChecked():
                self.wg.changeAxisBtn1.toggle()
                self.changeAxis()

            save_fname = QFileDialog.getSaveFileName(self, "Save Masks as bin", './untitled.bin')[0]
            if len(save_fname) < 1: 
                return
            if '.bin' not in save_fname:
                save_fname = save_fname + '.bin'

            for i in range(self.mask_arrList.shape[0]):
                fname = save_fname[:-4] + '_{}'.format(i+1) + save_fname[-4:]
                self.py_raw.m_Voxel = self.mask_arrList[i]
                self.py_raw.WriteToBin(fname)
                # self.py_raw.SaveWithoutHeader(fname)

            QMessageBox.information(self, 'Save All Masks', "All masks is saved.", \
                    QMessageBox.Ok, QMessageBox.Ok)
        except:
            print('saveMasksAsBin Error')

    def loadMasksNpy(self):
        try:
            if not self.is_opened:
                QMessageBox.warning(self, "Warining" ,"Image file is Not Opened.")
                return
            if not self.wg.changeAxisBtn1.isChecked():
                self.wg.changeAxisBtn1.toggle()
                self.changeAxis()

            mask_path = QFileDialog.getOpenFileName(self, 'Load Masks From Npy File', './')[0]
            mask_arr = np.load(mask_path)
            if self.NofI == mask_arr.shape[0]:
                self.mask_arrList[self.wg.maskComboBox.currentIndex()] = \
                    np.where(mask_arr.copy() > 0, 1, 0)
                self.mask_fname = mask_path.split('/')[-1]
                self.wg.lbl_mask_fname.setText('Mask file name : ' + self.mask_fname)
                self.refresh()
                QMessageBox.information(self, 'Load All Masks', "All masks is loaded.", \
                    QMessageBox.Ok, QMessageBox.Ok)
            else:
                print('loadNpyMasks Error : Mask volume and Image volume are different.')
        except:
            print('loadMasksNpy Error')

    def loadBinMasks(self):
        try:
            if not self.is_opened:
                QMessageBox.warning(self, "Warining" ,"Image file is Not Opened.")
                return
            if not self.wg.changeAxisBtn1.isChecked():
                self.wg.changeAxisBtn1.toggle()
                self.changeAxis()

            mask_path = QFileDialog.getOpenFileName(self, 'Load Masks From Bin File', './')[0]
            self.py_raw.ReadFromBin(mask_path)

            if self.NofI == self.py_raw.m_Voxel.shape[0]:
                self.mask_arrList[self.wg.maskComboBox.currentIndex()] = \
                    np.where(self.py_raw.m_Voxel.copy() > 0, 1, 0)
                self.mask_fname = mask_path.split('/')[-1]
                self.wg.lbl_mask_fname.setText('Mask file name : ' + self.mask_fname)
                self.refresh()
                QMessageBox.information(self, 'Load All Masks', "All masks is loaded.", \
                    QMessageBox.Ok, QMessageBox.Ok)
            else:
                print('loadBinMasks Error : Mask volume and Image volume are different.')
        except:
            print('loadBinMasks Error')

    def changeAxis(self):
        if self.is_opened:
            self.cur_idx = 0
            if self.wg.changeAxisBtn1.isChecked():
                if self.cur_axis == 0:
                    return
                elif self.cur_axis == 1:
                    self.EntireImage = np.transpose(self.EntireImage, (2, 0, 1))
                    self.mask_arrList = np.transpose(self.mask_arrList, (0, 3, 1, 2))
                elif self.cur_axis == 2:
                    self.EntireImage = np.transpose(self.EntireImage, (1, 2, 0))
                    self.mask_arrList = np.transpose(self.mask_arrList, (0, 2, 3, 1))
                self.cur_axis = 0
            elif self.wg.changeAxisBtn2.isChecked():
                if self.cur_axis == 0:
                    self.EntireImage = np.transpose(self.EntireImage, (1, 2, 0))
                    self.mask_arrList = np.transpose(self.mask_arrList, (0, 2, 3, 1))
                elif self.cur_axis == 1:
                    return
                elif self.cur_axis == 2:
                    self.EntireImage = np.transpose(self.EntireImage, (2, 0, 1))
                    self.mask_arrList = np.transpose(self.mask_arrList, (0, 3, 1, 2))
                self.cur_axis = 1
            elif self.wg.changeAxisBtn3.isChecked():
                if self.cur_axis == 0:
                    self.EntireImage = np.transpose(self.EntireImage, (2, 0, 1))
                    self.mask_arrList = np.transpose(self.mask_arrList, (0, 3, 1, 2))
                elif self.cur_axis == 1:
                    self.EntireImage = np.transpose(self.EntireImage, (1, 2, 0))
                    self.mask_arrList = np.transpose(self.mask_arrList, (0, 2, 3, 1))
                elif self.cur_axis == 2:
                    return
                self.cur_axis = 2
            
            self.NofI = self.EntireImage.shape[0]
            self.Nx = self.EntireImage.shape[1]
            self.Ny = self.EntireImage.shape[2]
            self.refresh()

    def showAllMasks(self, state):
        if state == Qt.Checked:
            r_img_arr = np.array([[[255, 0, 0, math.floor(255*self.transparent)]] * self.Ny] * self.Nx)
            g_img_arr = np.array([[[0, 255, 0, math.floor(255*self.transparent)]] * self.Ny] * self.Nx)
            b_img_arr = np.array([[[0, 0, 255, math.floor(255*self.transparent)]] * self.Ny] * self.Nx)

            sum_mask = np.zeros((self.Nx, self.Ny))
            for i in range(self.mask_arrList.shape[0]):
                sum_mask =  sum_mask + self.mask_arrList[i, self.cur_idx]

            r_label = np.where(sum_mask == 1, 1, 0).copy().reshape(self.Nx, self.Ny, 1)
            g_label = np.where(sum_mask == 2, 1, 0).copy().reshape(self.Nx, self.Ny, 1)
            b_label = np.where(sum_mask == 3, 1, 0).copy().reshape(self.Nx, self.Ny, 1)

            new_img = np.multiply(r_img_arr, r_label) + np.multiply(g_img_arr, g_label) + np.multiply(b_img_arr, b_label)
            self.wg.scene_1.addPixmap(QPixmap.fromImage(QImage(qimage2ndarray.array2qimage(new_img))))
        else:
            self.wg.scene_1.removeItem(self.wg.scene_1.items()[0])
            
    def morphBtn_clicked(self):
        if self.is_opened:
            img = np.array(self.mask_arrList)
            kernel = np.ones((3,3), np.uint8)

            processed = []
            for i in range(self.NofI):
                msk = img[self.wg.maskComboBox.currentIndex(), i, :, :]
                dilation = cv2.dilate(msk, kernel, iterations=1)
                erosion = cv2.erode(dilation, kernel, iterations=1)

                processed.append(erosion)

            processed = np.array(processed)
            self.mask_arrList[self.wg.maskComboBox.currentIndex()] = processed
            self.refresh()

    def predict_currentImg(self, state):
        if self.is_opened:
            if state == Qt.Checked:
                if not self.is_predicted:
                    img = self.EntireImage.copy()
                    img = img.reshape(-1, img.shape[1], img.shape[2], 1)

                    model = ym_model_1.build_unet()
                    model.load_weights('model/13_0.0079.h5')

                    prediction = model.predict(img)
                    prediction = np.squeeze(np.argmax(prediction, axis=-1))
                    self.pred_win.pred_volume = prediction
                    self.is_predicted = True
                self.pred_win.cur_idx = self.cur_idx
                self.pred_win.cur_image = self.cur_image
                self.pred_win.transparent = self.transparent
                self.pred_win.refresh()
                if not self.pred_win.isVisible(): self.pred_win.show()
            else:
                self.pred_win.close()
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())