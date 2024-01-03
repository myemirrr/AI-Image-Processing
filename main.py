import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5 import QtWidgets, QtGui, QtCore
import cv2 as cv
import cv2
import numpy as np
from HomepageUI import Ui_MainWindow


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        
        self.ui.imageAddButton.clicked.connect(self.openImageDialog)
        self.ui.bulaniklastirmaButton.clicked.connect(self.applyBlur)
        self.ui.histogramButton.clicked.connect(self.applyHistogram)
        self.ui.keskinlestirmeButton.clicked.connect(self.applySharpen)
        self.ui.trasholdingButton.clicked.connect(self.applyThresholding)  
        self.ui.otsutrasholdingButton.clicked.connect(self.applyOtsuThresholding) 
        self.ui.gamaButton.clicked.connect(self.applyGammaCorrection)  
        self.ui.pushButton_19.clicked.connect(self.applyCannyEdgeDetection) 
        self.ui.pushButton_20.clicked.connect(self.applyHarrisCornerDetection) 
        self.ui.pushButton_21.clicked.connect(self.applyCornerDetectionWithContours) 
        self.ui.resimdisikenarlikButton.clicked.connect(self.applyOuterEdgeDetection) 
        self.ui.dericheButton.clicked.connect(self.applyDericheEdgeDetection)
        self.ui.cannykenarButton.clicked.connect(self.applySobelEdgeDetection)

       
        self.ui.imageClear = QtWidgets.QLabel(self.ui.centralwidget)
        self.ui.imageClear.setGeometry(QtCore.QRect(486, 330, 281, 141))
        self.ui.imageClear.setFrameShape(QtWidgets.QFrame.Box)
        self.ui.imageClear.setText("")
        self.ui.imageClear.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.imageClear.setObjectName("imageClear")

        
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
            
            self.showImage(fileName)

    def showImage(self, imagePath):
        input_image = cv.imread(imagePath)
        self.original_image = input_image.copy()

        height, width, channels = input_image.shape
        bytesPerLine = channels * width
        qImg = QtGui.QImage(input_image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap_image = QtGui.QPixmap.fromImage(qImg)

        
        self.ui.imageUpload.setPixmap(pixmap_image.scaled(self.ui.imageUpload.width(), self.ui.imageUpload.height(), QtCore.Qt.KeepAspectRatio))
        self.ui.imageUpload.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.imageUpload.setScaledContents(True)

    def applyBlur(self):
        if self.original_image is not None:
            
            blurred_img = cv.GaussianBlur(self.original_image, (5, 5), 0)
            self.blurred_image = blurred_img.copy()

           
            height, width, channel = blurred_img.shape
            bytesPerLine = 3 * width
            qImg_blurred = QtGui.QImage(blurred_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            pixmap_blurred = QtGui.QPixmap.fromImage(qImg_blurred)

            
            self.ui.imageClear.setPixmap(pixmap_blurred.scaled(self.ui.imageClear.width(), self.ui.imageClear.height(), QtCore.Qt.KeepAspectRatio))
            self.ui.imageClear.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.imageClear.setScaledContents(True)
        else:
            print("Lütfen önce bir resim ekleyin.")

    def applyHistogram(self):
        if self.original_image is not None:
            
            gray_img = cv.cvtColor(self.original_image, cv.COLOR_BGR2GRAY)

            
            equalized_img = cv.equalizeHist(gray_img)
            self.equalized_image = equalized_img.copy()

            
            height, width = equalized_img.shape
            bytesPerLine = 1 * width
            qImg_equalized = QtGui.QImage(equalized_img.data, width, height, bytesPerLine, QtGui.QImage.Format_Grayscale8)
            pixmap_equalized = QtGui.QPixmap.fromImage(qImg_equalized)

            
            self.ui.imageClear.setPixmap(pixmap_equalized.scaled(self.ui.imageClear.width(), self.ui.imageClear.height(), QtCore.Qt.KeepAspectRatio))
            self.ui.imageClear.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.imageClear.setScaledContents(True)
        else:
            print("Lütfen önce bir resim ekleyin.")

    def applySharpen(self):
        if self.original_image is not None:
            
            sharpened_img = cv.filter2D(self.original_image, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
            self.sharpened_image = sharpened_img.copy()

            
            height, width, channel = sharpened_img.shape
            bytesPerLine = 3 * width
            qImg_sharpened = QtGui.QImage(sharpened_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            pixmap_sharpened = QtGui.QPixmap.fromImage(qImg_sharpened)

           
            self.ui.imageClear.setPixmap(pixmap_sharpened.scaled(self.ui.imageClear.width(), self.ui.imageClear.height(), QtCore.Qt.KeepAspectRatio))
            self.ui.imageClear.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.imageClear.setScaledContents(True)
        else:
            print("Lütfen önce bir resim ekleyin.")

    def applyThresholding(self):
        if self.original_image is not None:
            
            gray_img = cv.cvtColor(self.original_image, cv.COLOR_BGR2GRAY)

            
            _, thresholded_img = cv.threshold(gray_img, 127, 255, cv.THRESH_BINARY)
            self.thresholded_image = thresholded_img.copy()

           
            height, width = thresholded_img.shape
            bytesPerLine = 1 * width
            qImg_thresholded = QtGui.QImage(thresholded_img.data, width, height, bytesPerLine, QtGui.QImage.Format_Grayscale8)
            pixmap_thresholded = QtGui.QPixmap.fromImage(qImg_thresholded)

            
            self.ui.imageClear.setPixmap(pixmap_thresholded.scaled(self.ui.imageClear.width(), self.ui.imageClear.height(), QtCore.Qt.KeepAspectRatio))
            self.ui.imageClear.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.imageClear.setScaledContents(True)
        else:
            print("Lütfen önce bir resim ekleyin.")

    def applyOtsuThresholding(self):
        if self.original_image is not None:
            
            gray_img = cv.cvtColor(self.original_image, cv.COLOR_BGR2GRAY)

            
            _, otsu_thresholded_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            self.otsu_thresholded_image = otsu_thresholded_img.copy()

           
            height, width = otsu_thresholded_img.shape
            bytesPerLine = 1 * width
            qImg_otsu_thresholded = QtGui.QImage(otsu_thresholded_img.data, width, height, bytesPerLine, QtGui.QImage.Format_Grayscale8)
            pixmap_otsu_thresholded = QtGui.QPixmap.fromImage(qImg_otsu_thresholded)

            
            self.ui.imageClear.setPixmap(pixmap_otsu_thresholded.scaled(self.ui.imageClear.width(), self.ui.imageClear.height(), QtCore.Qt.KeepAspectRatio))
            self.ui.imageClear.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.imageClear.setScaledContents(True)
        else:
            print("Lütfen önce bir resim ekleyin.")

    def applyGammaCorrection(self):
        gamma = 1.5 
        if self.original_image is not None:
           
            gamma_corrected_img = np.power(self.original_image / float(np.max(self.original_image)), gamma)
            gamma_corrected_img = (gamma_corrected_img * 255).astype(np.uint8)
            self.gamma_corrected_image = gamma_corrected_img.copy()

            
            height, width, channel = gamma_corrected_img.shape
            bytesPerLine = 3 * width
            qImg_gamma_corrected = QtGui.QImage(gamma_corrected_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            pixmap_gamma_corrected = QtGui.QPixmap.fromImage(qImg_gamma_corrected)

            
            self.ui.imageClear.setPixmap(pixmap_gamma_corrected.scaled(self.ui.imageClear.width(), self.ui.imageClear.height(), QtCore.Qt.KeepAspectRatio))
            self.ui.imageClear.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.imageClear.setScaledContents(True)
        else:
            print("Lütfen önce bir resim ekleyin.")

            

    def applyCannyEdgeDetection(self):
     if self.original_image is not None:
        
        low_threshold = 50
        high_threshold = 150

        
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)

       
        edges = cv2.Canny(gray_image, low_threshold, high_threshold)

        
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        
        height, width, channel = edges_rgb.shape
        bytesPerLine = 3 * width
        qImg_edges = QtGui.QImage(edges_rgb.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap_edges = QtGui.QPixmap.fromImage(qImg_edges)

        self.ui.imageClear.setPixmap(pixmap_edges.scaled(self.ui.imageClear.width(), self.ui.imageClear.height(), QtCore.Qt.KeepAspectRatio))
        self.ui.imageClear.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.imageClear.setScaledContents(True)
     else:
      print("Lütfen önce bir resim ekleyin.")
      

    def applyHarrisCornerDetection(self):
      if self.original_image is not None:
        
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)

        
        corners = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)

        
        self.original_image[corners > 0.01 * corners.max()] = [0, 0, 255]  

        
        corners_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

        
        height, width, channel = corners_rgb.shape
        bytesPerLine = 3 * width
        qImg_corners = QtGui.QImage(corners_rgb.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap_corners = QtGui.QPixmap.fromImage(qImg_corners)

        self.ui.imageClear.setPixmap(pixmap_corners.scaled(self.ui.imageClear.width(), self.ui.imageClear.height(), QtCore.Qt.KeepAspectRatio))
        self.ui.imageClear.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.imageClear.setScaledContents(True)
      else:
        print("Lütfen önce bir resim ekleyin.")
       

    def applyCornerDetectionWithContours(self):
     if self.original_image is not None:
       
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)

        
        edges = cv2.Canny(gray_image, 50, 150)

        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

       
        image_with_contours = self.original_image.copy()
        cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)  

       
        image_with_contours_rgb = cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB)

        
        height, width, channel = image_with_contours_rgb.shape
        bytesPerLine = 3 * width
        qImg_with_contours = QtGui.QImage(image_with_contours_rgb.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap_with_contours = QtGui.QPixmap.fromImage(qImg_with_contours)

        self.ui.imageClear.setPixmap(pixmap_with_contours.scaled(self.ui.imageClear.width(), self.ui.imageClear.height(), QtCore.Qt.KeepAspectRatio))
        self.ui.imageClear.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.imageClear.setScaledContents(True)
     else:
        print("Lütfen önce bir resim ekleyin.")



    def applyOuterEdgeDetection(self):
      if self.original_image is not None:
       
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)

      
        edges = cv2.Canny(gray_image, 50, 150)

        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      
        black_image = np.zeros_like(self.original_image)

       
        cv2.drawContours(black_image, contours, -1, (255, 255, 255), 2)  

      
        result_image = cv2.bitwise_and(self.original_image, black_image)

        
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

      
        height, width, channel = result_image_rgb.shape
        bytesPerLine = 3 * width
        qImg_result = QtGui.QImage(result_image_rgb.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap_result = QtGui.QPixmap.fromImage(qImg_result)

        self.ui.imageClear.setPixmap(pixmap_result.scaled(self.ui.imageClear.width(), self.ui.imageClear.height(), QtCore.Qt.KeepAspectRatio))
        self.ui.imageClear.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.imageClear.setScaledContents(True)
      else:
        print("Lütfen önce bir resim ekleyin.")

  

    def applyDericheEdgeDetection(self):
     if self.original_image is not None:
        
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)

        
        deriche_x = cv2.ximgproc.GradientDericheX(gray_image, alpha=1.0, omega=1.0)
        deriche_y = cv2.ximgproc.GradientDericheY(gray_image, alpha=1.0, omega=1.0)
        
        
        edges = cv2.magnitude(deriche_x, deriche_y)

        
        _, binary_edges = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)

        
        binary_edges_rgb = cv2.cvtColor(binary_edges, cv2.COLOR_GRAY2RGB)

        
        height, width, channel = binary_edges_rgb.shape
        bytesPerLine = 3 * width
        qImg_binary_edges = QtGui.QImage(binary_edges_rgb.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap_binary_edges = QtGui.QPixmap.fromImage(qImg_binary_edges)

        self.ui.imageClear.setPixmap(pixmap_binary_edges.scaled(self.ui.imageClear.width(), self.ui.imageClear.height(), QtCore.Qt.KeepAspectRatio))
        self.ui.imageClear.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.imageClear.setScaledContents(True)
     else:
        print("Lütfen önce bir resim ekleyin.")

    def applySobelEdgeDetection(self):
     if self.original_image is not None:
       
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)

        
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        
        edges = cv2.magnitude(sobel_x, sobel_y)

       
        edges = np.uint8(edges)

       
        binary_edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        
        height, width, channel = binary_edges_rgb.shape
        bytesPerLine = 3 * width
        qImg_binary_edges = QtGui.QImage(binary_edges_rgb.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap_binary_edges = QtGui.QPixmap.fromImage(qImg_binary_edges)

        self.ui.imageClear.setPixmap(pixmap_binary_edges.scaled(self.ui.imageClear.width(), self.ui.imageClear.height(), QtCore.Qt.KeepAspectRatio))
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
