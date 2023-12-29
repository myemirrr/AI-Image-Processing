import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5 import QtWidgets, QtGui, QtCore
import cv2 as cv
import numpy as np
from HomepageUI import Ui_MainWindow

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Connect buttons to their respective functions
        self.ui.imageAddButton.clicked.connect(self.openImageDialog)
        self.ui.bulaniklastirmaButton.clicked.connect(self.applyBlur)
        self.ui.histogramButton.clicked.connect(self.applyHistogram)
        self.ui.keskinlestirmeButton.clicked.connect(self.applySharpen)
        self.ui.trasholdingButton.clicked.connect(self.applyThresholding)  
        self.ui.otsutrasholdingButton.clicked.connect(self.applyOtsuThresholding) 
        self.ui.gamaButton.clicked.connect(self.applyGammaCorrection)  


        # QLabel to show the image
        self.ui.imageClear = QtWidgets.QLabel(self.ui.centralwidget)
        self.ui.imageClear.setGeometry(QtCore.QRect(486, 330, 281, 141))
        self.ui.imageClear.setFrameShape(QtWidgets.QFrame.Box)
        self.ui.imageClear.setText("")
        self.ui.imageClear.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.imageClear.setObjectName("imageClear")

        # Variables to store original, blurred, equalized, sharpened, thresholded, and Otsu thresholded images
        self.original_image = None
        self.blurred_image = None
        self.equalized_image = None
        self.sharpened_image = None
        self.thresholded_image = None
        self.otsu_thresholded_image = None
        self.gamma_corrected_image = None
        self.deriche_edge_detected_image = None

    def openImageDialog(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.ReadOnly
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Resim Seç", "", "Resim Dosyaları (*.png; *.jpg; *.jpeg; *.bmp);;Tüm Dosyalar (*)", options=options)

        if fileName:
            # Show the selected image
            self.showImage(fileName)

    def showImage(self, imagePath):
        input_image = cv.imread(imagePath)
        self.original_image = input_image.copy()

        height, width, channels = input_image.shape
        bytesPerLine = channels * width
        qImg = QtGui.QImage(input_image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap_image = QtGui.QPixmap.fromImage(qImg)

        # Display the image in the imageUpload QLabel
        self.ui.imageUpload.setPixmap(pixmap_image.scaled(self.ui.imageUpload.width(), self.ui.imageUpload.height(), QtCore.Qt.KeepAspectRatio))
        self.ui.imageUpload.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.imageUpload.setScaledContents(True)

    def applyBlur(self):
        if self.original_image is not None:
            # Apply a blur effect using OpenCV directly on the original image
            blurred_img = cv.GaussianBlur(self.original_image, (5, 5), 0)
            self.blurred_image = blurred_img.copy()

            # Convert the numpy array to QPixmap and update the displayed image with the blurred version
            height, width, channel = blurred_img.shape
            bytesPerLine = 3 * width
            qImg_blurred = QtGui.QImage(blurred_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            pixmap_blurred = QtGui.QPixmap.fromImage(qImg_blurred)

            # Display the blurred image in the imageClear QLabel
            self.ui.imageClear.setPixmap(pixmap_blurred.scaled(self.ui.imageClear.width(), self.ui.imageClear.height(), QtCore.Qt.KeepAspectRatio))
            self.ui.imageClear.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.imageClear.setScaledContents(True)
        else:
            print("Lütfen önce bir resim ekleyin.")

    def applyHistogram(self):
        if self.original_image is not None:
            # Convert the image to grayscale
            gray_img = cv.cvtColor(self.original_image, cv.COLOR_BGR2GRAY)

            # Apply histogram equalization
            equalized_img = cv.equalizeHist(gray_img)
            self.equalized_image = equalized_img.copy()

            # Convert the numpy array to QPixmap and update the displayed image with the equalized version
            height, width = equalized_img.shape
            bytesPerLine = 1 * width
            qImg_equalized = QtGui.QImage(equalized_img.data, width, height, bytesPerLine, QtGui.QImage.Format_Grayscale8)
            pixmap_equalized = QtGui.QPixmap.fromImage(qImg_equalized)

            # Display the equalized image in the imageClear QLabel
            self.ui.imageClear.setPixmap(pixmap_equalized.scaled(self.ui.imageClear.width(), self.ui.imageClear.height(), QtCore.Qt.KeepAspectRatio))
            self.ui.imageClear.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.imageClear.setScaledContents(True)
        else:
            print("Lütfen önce bir resim ekleyin.")

    def applySharpen(self):
        if self.original_image is not None:
            # Apply sharpening using OpenCV directly on the original image
            sharpened_img = cv.filter2D(self.original_image, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
            self.sharpened_image = sharpened_img.copy()

            # Convert the numpy array to QPixmap and update the displayed image with the sharpened version
            height, width, channel = sharpened_img.shape
            bytesPerLine = 3 * width
            qImg_sharpened = QtGui.QImage(sharpened_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            pixmap_sharpened = QtGui.QPixmap.fromImage(qImg_sharpened)

            # Display the sharpened image in the imageClear QLabel
            self.ui.imageClear.setPixmap(pixmap_sharpened.scaled(self.ui.imageClear.width(), self.ui.imageClear.height(), QtCore.Qt.KeepAspectRatio))
            self.ui.imageClear.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.imageClear.setScaledContents(True)
        else:
            print("Lütfen önce bir resim ekleyin.")

    def applyThresholding(self):
        if self.original_image is not None:
            # Convert the image to grayscale
            gray_img = cv.cvtColor(self.original_image, cv.COLOR_BGR2GRAY)

            # Apply thresholding
            _, thresholded_img = cv.threshold(gray_img, 127, 255, cv.THRESH_BINARY)
            self.thresholded_image = thresholded_img.copy()

            # Convert the numpy array to QPixmap and update the displayed image with the thresholded version
            height, width = thresholded_img.shape
            bytesPerLine = 1 * width
            qImg_thresholded = QtGui.QImage(thresholded_img.data, width, height, bytesPerLine, QtGui.QImage.Format_Grayscale8)
            pixmap_thresholded = QtGui.QPixmap.fromImage(qImg_thresholded)

            # Display the thresholded image in the imageClear QLabel
            self.ui.imageClear.setPixmap(pixmap_thresholded.scaled(self.ui.imageClear.width(), self.ui.imageClear.height(), QtCore.Qt.KeepAspectRatio))
            self.ui.imageClear.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.imageClear.setScaledContents(True)
        else:
            print("Lütfen önce bir resim ekleyin.")

    def applyOtsuThresholding(self):
        if self.original_image is not None:
            # Convert the image to grayscale
            gray_img = cv.cvtColor(self.original_image, cv.COLOR_BGR2GRAY)

            # Apply Otsu's thresholding
            _, otsu_thresholded_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            self.otsu_thresholded_image = otsu_thresholded_img.copy()

            # Convert the numpy array to QPixmap and update the displayed image with the Otsu thresholded version
            height, width = otsu_thresholded_img.shape
            bytesPerLine = 1 * width
            qImg_otsu_thresholded = QtGui.QImage(otsu_thresholded_img.data, width, height, bytesPerLine, QtGui.QImage.Format_Grayscale8)
            pixmap_otsu_thresholded = QtGui.QPixmap.fromImage(qImg_otsu_thresholded)

            # Display the Otsu thresholded image in the imageClear QLabel
            self.ui.imageClear.setPixmap(pixmap_otsu_thresholded.scaled(self.ui.imageClear.width(), self.ui.imageClear.height(), QtCore.Qt.KeepAspectRatio))
            self.ui.imageClear.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.imageClear.setScaledContents(True)
        else:
            print("Lütfen önce bir resim ekleyin.")

    def applyGammaCorrection(self):
        gamma = 1.5  # You can adjust the gamma value
        if self.original_image is not None:
            # Apply gamma correction
            gamma_corrected_img = np.power(self.original_image / float(np.max(self.original_image)), gamma)
            gamma_corrected_img = (gamma_corrected_img * 255).astype(np.uint8)
            self.gamma_corrected_image = gamma_corrected_img.copy()

            # Convert the numpy array to QPixmap and update the displayed image with the gamma-corrected version
            height, width, channel = gamma_corrected_img.shape
            bytesPerLine = 3 * width
            qImg_gamma_corrected = QtGui.QImage(gamma_corrected_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            pixmap_gamma_corrected = QtGui.QPixmap.fromImage(qImg_gamma_corrected)

            # Display the gamma-corrected image in the imageClear QLabel
            self.ui.imageClear.setPixmap(pixmap_gamma_corrected.scaled(self.ui.imageClear.width(), self.ui.imageClear.height(), QtCore.Qt.KeepAspectRatio))
            self.ui.imageClear.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.imageClear.setScaledContents(True)
        else:
            print("Lütfen önce bir resim ekleyin.")

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
