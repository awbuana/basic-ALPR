import numpy as np
import cv2
from matplotlib import pyplot as plt
from operator import itemgetter

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*28*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(28, 28),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
    return img

def number_segmentation(img_rgb):
    
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    #gray = cv2.bilateralFilter(gray,9,75,75)
    #gray = cv2.equalizeHist(gray)
    #gray = img_rgb
    #clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(2,8))
    #gray = clahe.apply(gray)
    ret, gray= cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img = gray
    
    cv2.namedWindow("threshold seg",cv2.WINDOW_NORMAL)
    image,cnts,hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #bener

    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:15]


    lebar_kotak=[]
    tinggi_kotak=[]
    area_kotak=[]
    
    x_w_y_h_kotak=[]


    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        x_w_y_h_kotak.append((x,w,y,h))
        
    # lebar_kotak = sorted(lebar_kotak)
    # tinggi_kotak = sorted(tinggi_kotak)
    # mean_lebar_kotak = np.mean(lebar_kotak)
    # mean_tinggi_kotak =  np.mean(tinggi_kotak)
    # mean_area_kotak = np.mean(area_kotak)
    # std_area_kotak = np.mean(area_kotak)
    # std_lebar_kotak = np.std(lebar_kotak)
    # std_tinggi_kotak = np.std(tinggi_kotak)   
    
    count_kotak = 0
    temp_x = -1
    temp_y = -1

    # urutin kotak berdasarkan x
    x_w_y_h_kotak.sort(key=itemgetter(0))
    filtered_kotak=[]
    for i in range(len(x_w_y_h_kotak)):
        x,w,y,h = x_w_y_h_kotak[i]
        dempet = False
        kotak_didalem = False
        # ilangin kotak dempet
        if(i>0):
            xx,ww,yy,hh = x_w_y_h_kotak[i-1]
            if(abs(xx-x)<=1 and ww>w):
                #print(str(xx)+" "+str(x))
                dempet=True
        jumlah_kotak_valid = len(filtered_kotak)
        # buang kotak didalem
        if(jumlah_kotak_valid>0):
            xx,ww,yy,hh = filtered_kotak[jumlah_kotak_valid-1]
            #print(str(x) + " " + str(xx)+" "+str(x>=xx and x+w<=xx+ww))
            if(x>=xx and x+w<=xx+ww and ww<=hh):
                kotak_didalem = True
        # buang kotak didalem
        if(jumlah_kotak_valid>1):
            xx,ww,yy,hh = filtered_kotak[jumlah_kotak_valid-2]
            #print(str(x+w) + " " + str(xx+xx)+" "+str(x>=xx and x+w<=xx+ww))
            if(x>=xx and x+w<=xx+ww and ww<=hh):
                kotak_didalem = True
        # buang kotak didalem
        if(jumlah_kotak_valid>2):
            xx,ww,yy,hh = filtered_kotak[jumlah_kotak_valid-3]
            #print(str(x+w) + " " + str(xx+xx)+" "+str(x>=xx and x+w<=xx+ww))
            if(x>=xx and x+w<=xx+ww and ww<=hh):
                kotak_didalem = True
        # if((abs(w - mean_lebar_kotak) < (1.5) * std_lebar_kotak) and (abs(h - mean_tinggi_kotak) < (1.5) * std_tinggi_kotak) 
        # and (abs((w*h) - mean_area_kotak) < (0.8) * std_area_kotak) and (temp_y != y and temp_x !=x)
        # ):
        if(w<h and not dempet and not kotak_didalem):
            #print(str(x) + " " + str(xx)+" "+str(x>xx and h<hh))
            # cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),1)
            # cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(0,0,255),1)
            filtered_kotak.append((x,w,y,h))
            # belakangan
            tinggi_kotak.append(h)


    # ambil 8 pertama secara x
    if(len(filtered_kotak)>8):
       filtered_kotak = filtered_kotak[:8]
       tinggi_kotak = tinggi_kotak[:8]

   
    mean_tinggi_kotak = np.mean(tinggi_kotak)
    std_tinggi_kotak = np.std(tinggi_kotak)
    filtered_2_kotak=[]
    # seleksi kotak yang yang outlier
    for i in range(len(filtered_kotak)):
        x,w,y,h = filtered_kotak[i]
        #print(str(w*h) + " "+ str(mean_tinggi_kotak))
        if(h>=0.9*mean_tinggi_kotak or (abs(h - mean_tinggi_kotak) < (0.7) * std_tinggi_kotak)):
            filtered_2_kotak.append((x,w,y,h))

    diff_x_kotak=[]
    
    
    # cari jarak antar 2 kotak terdekat (mau ambil angka aja)
    for i in range(len(filtered_2_kotak)-1):
        x,w,y,h = filtered_2_kotak[i]
        xx,ww,yy,hh = filtered_2_kotak[i+1]
        diff_x_kotak.append(abs(xx-(x+w)))

    # print("diff")
    # print(diff_x_kotak)
    # print(np.mean(diff_x_kotak))
    # print(filtered_2_kotak) 

    mean_diff_x_kotak = np.mean(diff_x_kotak)   

    angka_kotak=[]
    batas_x_huruf=0
    # nyari angka (dari kotak yang deket deket)
    for i in range(len(diff_x_kotak)):
        x,w,y,h =filtered_2_kotak[i]
       
        #print(str(diff_x_kotak[i]) + " "+ str(mean_diff_x_kotak))
        if(diff_x_kotak[i]<mean_diff_x_kotak):
            angka_kotak.append((x,w,y,h))
            count_kotak+=1
        else:
            if(i!=0 and i!=len(diff_x_kotak)-1):
                angka_kotak.append((x,w,y,h))
                count_kotak+=1
            elif(i!=0 and i==len(diff_x_kotak)-1):
                print("batas arr terakhir")
                angka_kotak.append((x,w,y,h))
                count_kotak+=1
                
                if(len(angka_kotak)<4):
                    xx,ww,yy,hh =filtered_2_kotak[i+1]
                    angka_kotak.append((xx,ww,yy,hh))
                    count_kotak+=1
            
            batas_x_huruf+=1
            if(len(angka_kotak)>1 and i!=len(diff_x_kotak)-1):
                break
            

        if(batas_x_huruf==2 or len(angka_kotak)==4):
            break
    
    list_angka=[]

    cv2.drawContours(img_rgb, cnts, -1, (0, 255, 0), 1)
    cv2.imshow("threshold seg", img_rgb)
    
    
    # kumpulin list gambar angka sama save angka (debug) 
    for i in range(len(angka_kotak)):
        x,w,y,h =angka_kotak[i]
        # cv2.namedWindow("image plat"+str(i),cv2.WINDOW_NORMAL)
        # cv2.imshow("image plat"+str(i),img[y:y+h, x:x+w])
        # cv2.waitKey()
        crop = img[(y-1):(y+h+2), (x-1):(x+w+1)]
    

        print "#"*10
        print crop.shape
        row, col = crop.shape
        npad = int(row/10)
        pad = np.zeros((npad,col), np.uint8)
        crop = np.vstack((crop,pad))
        crop = np.vstack((pad,crop))
        row, col = crop.shape
        npad = int((row-col)/2)
        pad = np.zeros((row,npad), np.uint8)
        crop = np.hstack((crop,pad))
        crop = np.hstack((pad,crop))
        print crop.shape
        

        res = cv2.resize(crop,(28,28), interpolation = cv2.INTER_AREA)
        print res.shape

        kernel = np.ones((2,2), np.uint8)
        res = cv2.dilate(res,kernel,iterations = 2)

        kernel = np.ones((2,2), np.uint8)
        res = cv2.erode(res,kernel,iterations = 3)

        kernel = np.ones((5,5), np.uint8)
        res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
        
        print "#"*10
        # kernel = np.ones((1,1), np.uint8)
        # res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
        # kernel = np.ones((2,2), np.uint8)
        # res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
        
        res = deskew(res)
        cv2.imwrite("angka-" + str(i) + ".jpg", res)
        list_angka.append(res)
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),1)
        cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(0,0,255),1)

    #print(img.shape)
    #print((list_angka[0]).shape)
    print(str(count_kotak) +"kotak")


    cv2.imshow("img black seg",img)
    cv2.imshow("img rgb seg",img_rgb)
    
    cv2.waitKey(0)

    return list_angka

