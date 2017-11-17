import cv2
# Importing the Opencv Library
import numpy as np
# Importing NumPy,which is the fundamental package for scientific computing with Python
from matplotlib import pyplot as plt
import sys
from segmentation import number_segmentation
from predict import predict_plate


# Reading Image
img = cv2.imread(sys.argv[1])
print(img.shape)
cv2.namedWindow("Original Image",cv2.WINDOW_NORMAL)
# Creating a Named window to display image
cv2.imshow("Original Image",img)
# Display image

def preprocess_image(img):
    # RGB to Gray scale conversion
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)


    # Noise removal with iterative bilateral filter(removes noise while preserving edges)
    noise_removal = cv2.bilateralFilter(img_gray,9,75,75)

    equal_histogram = noise_removal


    # Morphological opening with a rectangular structure element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=3)
    #cv2.imshow("Morphological opening",morph_image)


    # Image subtraction(Subtracting the Morphed image from the histogram equalised Image)
    sub_morp_image = cv2.subtract(equal_histogram,morph_image)
    #cv2.imshow("Subtraction image", sub_morp_image)


    # Thresholding the image
    ret,thresh_image = cv2.threshold(sub_morp_image,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.namedWindow("Image after Thresholding",cv2.WINDOW_NORMAL)
    # Creating a Named window to display image
    cv2.imshow("Image after Thresholding",thresh_image)
    # Display Image

    #bersih bercak bercak
    kernel = np.ones((2,2), np.uint8)
    eroded_image = cv2.erode(thresh_image,kernel,iterations = 1)
    #cv2.namedWindow("erosion", cv2.WINDOW_NORMAL)
    # Creating a Named window to display image
    #cv2.imshow("erosion", thresh_image)

    return thresh_image,eroded_image

def contouring_img(thresh_image):
    # Applying Canny Edge detection
    # canny_image = cv2.Canny(thresh_image,250,255)
    # cv2.namedWindow("Image after applying Canny",cv2.WINDOW_NORMAL)
    # # Creating a Named window to display image
    # cv2.imshow("Image after applying Canny",canny_image)


    # # dilation to strengthen the edges
    # kernel = np.ones((2,2), np.uint8)
    # dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
    # cv2.namedWindow("Dilation", cv2.WINDOW_NORMAL)
    # cv2.imshow("Dilation", dilated_image)
    # Displaying Image



    # Finding Contours in the image based on edges
    new,contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours= sorted(contours, key = cv2.contourArea, reverse = True)[:100]

    return contours



def get_data_rectangle(img,contours):
    lebar_kotak=[]
    tinggi_kotak=[]
    area_kotak=[]
    kumpulan_x=[]
    kumpulan_y=[]
    kumpulan_w=[]
    kumpulan_h=[]
    final = cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
    cnts = contours
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area_kotak.append(w*h)
        lebar_kotak.append(w)
        tinggi_kotak.append(h)

        area_kotak = sorted(area_kotak)
        lebar_kotak = sorted(lebar_kotak)
        tinggi_kotak = sorted(tinggi_kotak)
        mean_lebar_kotak = np.mean(lebar_kotak)
        mean_tinggi_kotak =  np.mean(tinggi_kotak)
        mean_area_kotak = np.mean(area_kotak)
        std_area_kotak = np.mean(area_kotak)
        std_lebar_kotak = np.std(lebar_kotak)
        std_tinggi_kotak = np.std(tinggi_kotak)

        count_kotak = 0
        temp_x = -1
        temp_y = -1
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)

            if((abs(w - mean_lebar_kotak) < (1.5) * std_lebar_kotak) and (abs(h - mean_tinggi_kotak) < (1.5) * std_tinggi_kotak) 
            and (abs((w*h) - mean_area_kotak) < (0.8) * std_area_kotak) and (temp_y != y and temp_x !=x)
            and w<h):
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
                count_kotak+=1
                kumpulan_y.append(y)
                kumpulan_x.append(x)
                kumpulan_w.append(w)
                kumpulan_h.append(h) 
                

            temp_x = x
            temp_y = y



    # Drawing the selected contour on the original image
    cv2.namedWindow("Image with Selected Contour",cv2.WINDOW_NORMAL)
    # Creating a Named window to display image
    cv2.imshow("Image with Selected Contour",final)

    return (kumpulan_x, kumpulan_y,kumpulan_w, kumpulan_h) 



#horizontal histograam ===================================
def horizontal_img_proc(thresh_image):
    new_image = thresh_image
    new_image = cv2.bilateralFilter(new_image,9,75,75)
    img_back = cv2.equalizeHist(new_image)

    image_data = np.asarray(img_back)

    hist_horizon = []
    for i in range(len(image_data[0])-1):
        val=0
        for j in range(len(image_data)-1):
            val += abs(image_data[j][i]-image_data[j+1][i])
            #print(image_data[i][j]-image_data[i+1][j+1])

        hist_horizon.append(val)
        #print(hist_horizon)

    mean_horizontal = np.mean(hist_horizon)  
    for i in range(len(hist_horizon)):
        if(hist_horizon[i]<mean_horizontal):
            hist_horizon[i]=0      

    xvalue =[]
    for i in range(len(hist_horizon)-1):
        if(hist_horizon[i]==0 and hist_horizon[i+1]!=0):
            for j in range(i+1,len(hist_horizon)-12):
                if(hist_horizon[j]==0 and hist_horizon[j+1]==0 and hist_horizon[j+2]==0
                and hist_horizon[j+3]==0 and hist_horizon[j+4]==0 and hist_horizon[j+5]==0
                and hist_horizon[j+6]==0 and hist_horizon[j+7]==0 and hist_horizon[j+8]==0
                and hist_horizon[j+11]==0):
                    xvalue.append((i,j))
                    i=j
                    break

    #plt.subplot(221), plt.plot(hist_horizon)
    #plt.subplot(222), plt.imshow(new_image)
    #plt.show()

    return xvalue
    #horizontal histograam ===================================

#vertical histograam ===================================
def vertical_img_proc(new_image):
    new_image = cv2.bilateralFilter(new_image,9,75,75)
    img_back = cv2.equalizeHist(new_image)
    image_data = np.asarray(img_back)

    hist_vertical = []
    for i in range(len(image_data)-1):
        val=0
        for j in range(len(image_data[0])-1):
            val += abs(image_data[i][j]-image_data[i][j+1])
            
        hist_vertical.append(val)
        #print(hist_horizon)

    mean_vertical = np.mean(hist_vertical)  
    for i in range(len(hist_vertical)):
        if(hist_vertical[i]<mean_vertical):
            hist_vertical[i]=0      

    yvalue =[]
    for i in range(len(hist_vertical)-1):
        if(hist_vertical[i]==0 and hist_vertical[i+1]!=0):
            for j in range(i+1,len(hist_vertical)-3):
                if(hist_vertical[j]==0 and hist_vertical[j+1]==0 and hist_vertical[j+2]==0
                ):
                    yvalue.append((i,j))
                    i=j
                    break


    # plt.subplot(221), plt.plot(hist_vertical)
    # plt.subplot(222), plt.imshow(new_image)
    # plt.show()

    return yvalue
    #vertical histograam ===================================

def crop_plat(kumpulan_x,kumpulan_y,kumpulan_w,kumpulan_h,xvalue,yvalue,img):    
    start_x = int(np.percentile(kumpulan_x,25))
    start_y = int(np.median(kumpulan_y))
    end_x = int(np.percentile(kumpulan_x,75))
    end_y = int(start_y + np.max(kumpulan_h))


    ### itung kotak  x wise
    sum_kotak=0
    arr_kotak=[]
    for i in range(len(xvalue)):
        for j in range(len(kumpulan_x)):
            a,b = xvalue[i]
            if(a<= kumpulan_x[j] and b>= kumpulan_x[j] ):
                sum_kotak+=1
        
        arr_kotak.append(sum_kotak)
        sum_kotak=0

    # print("x")
    # print(arr_kotak)
    # print(xvalue)
    # print(xvalue[np.argmax(arr_kotak)])

    ### itung kotak  y wise
    sum_kotak_y=0
    arr_kotak_y=[]
    for i in range(len(yvalue)):
        for j in range(len(kumpulan_y)):
            a,b = yvalue[i]
            if(a<= kumpulan_y[j] and b>= kumpulan_y[j] ):
                sum_kotak_y+=1
        
        arr_kotak_y.append(sum_kotak_y)
        sum_kotak_y=0

    # print("y")
    # print(arr_kotak_y)
    # print(yvalue[np.argmax(arr_kotak_y)])
    # print(yvalue)

    mask = np.zeros(img.shape[:2], np.uint8)
    a,b = xvalue[np.argmax(arr_kotak)]
    c,d = yvalue[np.argmax(arr_kotak_y)]
    
    masked_img = img[(c-5):(d+5), a-5:b+5]
    cv2.imshow('plat',masked_img)
    return masked_img
    #cv2.waitKey()

## MAIN
copy_img = img.copy()
thresh_image,eroded_image = preprocess_image(img)

cnts = contouring_img(eroded_image)

kumpulan_x, kumpulan_y,kumpulan_w, kumpulan_h = get_data_rectangle(img,cnts)

x_area = horizontal_img_proc(thresh_image)
y_area = vertical_img_proc(thresh_image)

cv2.imshow('sebelum di crop',copy_img)
#masked_img_binary = crop_plat(kumpulan_x,kumpulan_y,kumpulan_w,kumpulan_h,x_area,y_area,thresh_image)
masked_img_rgb = crop_plat(kumpulan_x,kumpulan_y,kumpulan_w,kumpulan_h,x_area,y_area,copy_img)
cv2.waitKey()

list_angka = number_segmentation(masked_img_rgb)

prediksi = predict_plate(list_angka)

print("nomor plat =" + prediksi)
