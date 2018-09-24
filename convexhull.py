from PIL import Image
import cv2
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image
from skimage import data, img_as_float
from scipy import misc

#from skimage.util import invert

# The original image is inverted as the object must be white.
#image = invert(data.horse())
# convert to binary after thresholding
img = cv2.imread('editted2.jpg',1)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
plt.imshow(thresh1,'gray')
plt.imsave("thresh.jpg",thresh1)
#conversion done
img1=cv2.imread('thresh_project.jpg',cv2.IMREAD_GRAYSCALE)
#ret,thresh2 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)
#img1=Image.open('thresh.jpg').convert('L')
rows,cols = img1.shape
for i in range(rows):
    for j in range(150):
        thresh1[i,j]=0
#plt.imshow(img1,'gray')
plt.imsave("thresh1.jpg",thresh1)
im = misc.imread('thresh1.jpg')
#print (im)
gray = im.sum(-1)


# convex hull image
chull = convex_hull_image(gray)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

ax[0].set_title('Original picture')
ax[0].imshow(img, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_axis_off()

ax[1].set_title('Transformed picture')
ax[1].imshow(chull, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_axis_off()

plt.imsave("thresh2.jpg",chull)
plt.imshow(chull, cmap=plt.cm.gray, interpolation='nearest')

plt.tight_layout()
plt.show()
# plotted and saved

#centre of gravity from original pixel by validating from
#segmented convex hull image
X=0
Y=0
P=0

implot = plt.imshow(img,'gray')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
for i in range(rows):
    for j in range(cols):
        if chull[i,j]:
            P=P+gray[i,j]
for i in range(rows):
    for j in range(cols):
        if chull[i,j]:
#            plt.plot(j, i,color="red")
#            plt.scatter(j, i, c='r')
            X=X+(gray[i,j]*i*1.0)
            Y=Y+(gray[i,j]*j*1.0)
X=X/P
Y=Y/P
X=int(X)
Y=int(Y)
print (X,Y,P,gray[Y,X])
ret,thresh3 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
plt.imshow(thresh3,'gray')
#im1 = plt.imread('editted1.jpg')
#implot = plt.imshow(im1)

# draw white line by editing the image then convert the image by using 
# threshold value segment the image


# put a blue dot at (10, 20)
#plt.scatter([10], [20])

# put a red dot, size 40, at 2 locations:
plt.scatter(304, 245, c='r') #plots single point
plt.scatter(306, 245, c='b')
plt.scatter(Y-20, X, c='g')
plt.show() #to plot single point

plt.imshow(thresh1,'gray')
circle1 =plt.Circle((Y, X), 80, color='r',fill=False)  #not working
fig = plt.gcf()
ax = fig.gca()
ax.add_artist(circle1)
fig.savefig('plotcircles.png')
plt.show()
#trying to segment the region using circle
#img = misc.imread('editted2.jpg')
#plt.imshow(img,'gray')
#img = cv2.circle(img, (100, 400), 2, (255,0,0), 3)
#plt.imshow(img,'gray')
#not  working

p1=[0,0]
p2=[0,0]
middle=[]
q=[]
visited=[]
visited.append([Y,X])
q.append([Y,X])

while len(q):
    k=q.pop()
    d=[[1,0],[-1,0],[-1,-1],[1,-1],[0,-1]]
    for i in range(0,5):
        a=k[0]+d[i][0]
        b=k[1]+d[i][1]
        if gray[a,b]<220 and gray[a,b]>180:
            if [a,b] not in visited:
                #print("hi")
                q.append([a,b])
                visited.append([a,b])
                if(a<Y and p1[1]<b):
                    p1[1]=b
                    p1[0]=a
                if(a>Y and p2[1]<b):
                    p2[1]=b
                    p2[0]=a

# thresholding to get corpus collosum
img = cv2.imread('editted2.jpg',cv2.IMREAD_GRAYSCALE)
print (gray[Y+20,X],gray[Y,X],gray[Y-20,X])
ret,thresh4 = cv2.threshold(img,img[Y,X],255,cv2.THRESH_BINARY)
plt.imshow(thresh4,'gray')
plt.show()
#not working