import numpy as np
import math
import cv2

#<- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ->
#<------------------------------ GAUSSIAN FUNCTION ---------------------------------->
def gaussian(x, sd):
    return (1/(sd*math.sqrt(2*math.pi)))*math.exp((-(x**2))/(2*(sd**2)))

#<------------- CALCULATE THE DISTANCE BETWEEN TWO PIXEL CO-ORDINATES --------------->
def distance(x,y,i,j):
    return math.sqrt(((x-i)**2)+((y-j)**2))

#<------------------------ APPLY THE JOINT BILATERAL FILTER ------------------------->
def jointBilFilter(imgA, imgF, sig_d, sig_r, diameter):
    image = np.ones(imgA.shape)                                                                     #create a new image that is the same size as the input image A (each pixel is initialised as [1,1,1]]
    rows = imgA.shape[0]                                                                            #store the number of rows in variable rows
    cols = imgA.shape[1]                                                                            #store the number of columns in variable cols

    if diameter%2 == 0:                                                                             #if the diameter provided is even
        radius = diameter/2                                                                         #divide by 2 to get the radius
    else:
        radius = (diameter-1)/2                                                                     #if the diameter is odd then subtract 1 and divide by 2 to get the radius
    radius = int(radius)                                                                            #convert the radius from a float to an int

    for y in range(rows):                                                                           #y represents the row number as it rows are stacked vertically
        for x in range(cols):                                                                       #x represents the column number as columns are stacked horizontally
            b,g,r = imgF.item(y,x,0), imgF.item(y,x,1), imgF.item(y,x,2)                            #individually reads the rgb bands of the current pixel x,y
            den_B = 0; den_G = 0; den_R = 0                                                         #as we apply the filter separately to each rgb band, we need to initialise each denominator at 1
            sum_B = 0; sum_G = 0; sum_R = 0                                                         #initialise the sum of the weighted neighborhood pixels at 0 for each rgb band

            for i in range(x-radius,x+radius+1):                                                    #iterates through x values in the neighborhood
                for j in range(y-radius,y+radius+1):                                                #iterates through y values within neighborhood
                    if i >= 0 and i < cols and j > 0 and j < rows:                                  #only consider pixels that are within the defined range / not out of bounds
                        gd = gaussian(distance(x,y,i,j), sig_d)                                     #calculate gd, the gaussian function of the spatial proximity between pixels at coordinates x,y and i,j

                        #<--- individually calculate gr for each rgb band --->                      
                        gausBLUE = gaussian(b - imgF.item(j,i,0), sig_r)
                        gausGREEN = gaussian(g - imgF.item(j,i,1), sig_r)
                        gausRED = gaussian(r - imgF.item(j,i,2), sig_r)

                        #<-- add pixel weight to denominator for each rgb band -->
                        den_B += gd*gausBLUE
                        den_G += gd*gausGREEN
                        den_R += gd*gausRED

                        #<--- add weighted intensity to sum for each rgb band --->
                        sum_B += imgA.item(j,i,0)*gd*gausBLUE
                        sum_G += imgA.item(j,i,1)*gd*gausGREEN
                        sum_R += imgA.item(j,i,2)*gd*gausRED

            #<--- update the pixel to it's new value in the new image --->
            image[y][x] = [(sum_B/den_B), (sum_G/den_G), (sum_R/den_R)]
    return image
#<- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ->

#<------------------ APPLY FILTER TO TEST IMAGES -----------------------> 
imgA = cv2.imread('test3a.jpg'); imgF = cv2.imread('test3b.jpg')                                    #read in the flash and no-flash images - note that the flash image must be imgF and the no-flash image must be imgA                 
sig_d = 5; sig_r = 5; diameter = 5                                                                  #define values for sigma d and sigma r and diameter
filteredImg = jointBilFilter(imgA, imgF, sig_d, sig_r, diameter)                                    #store the filtered image in the variable called filteredImg
cv2.imwrite(str(sig_d)+'-'+str(sig_r)+'-'+str(diameter)+'.jpg', filteredImg)                        #write the image to the current directory and name it based on the current parameters of the filter
#<---------------------------------------------------------------------->
