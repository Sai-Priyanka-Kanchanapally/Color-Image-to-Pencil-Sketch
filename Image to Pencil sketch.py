#!/usr/bin/env python
# coding: utf-8

# # Import the libraries

# In[11]:


get_ipython().system('pip install opencv-python')
get_ipython().system('pip install opencv-contrib-python')
import cv2
import matplotlib.pyplot as plt


# # Get the image file

# In[18]:


img_file = 'parrot.jpg'


# In[23]:


original_image = cv2.imread(img_file)


# # convert BGR image to RGB image

# In[24]:


original_img_rgb = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)


# # Display the original image

# In[25]:


plt.imshow(original_img_rgb)
plt.axis('off')
plt.title('Original Image')
plt.show()


# # Convert the image to grayscale

# In[26]:


gray_image = cv2.cvtColor(original_img_rgb,cv2.COLOR_BGR2GRAY)


# In[27]:


plt.imshow(gray_image)
plt.axis('off')
plt.title('GrayScale image')
plt.show()


# # Invert the grayscale image

# In[28]:


inverted_gray_image = cv2.bitwise_not(gray_image)


# In[29]:


plt.imshow(inverted_gray_image)
plt.axis('off')
plt.title('Inverted Gray Image')
plt.show()


# # Blur the inverted image using GaussianBlur functions

# In[31]:


blurred_image = cv2.GaussianBlur(inverted_gray_image,(111,111),0)


# In[32]:


plt.imshow(blurred_image)
plt.axis('off')
plt.title('BLurred Image')
plt.show()


# # Invert the blurred image

# In[33]:


inverted_blurred_image = cv2.bitwise_not(blurred_image)


# In[34]:


plt.imshow(inverted_blurred_image)
plt.axis('off')
plt.title('Inverted BLurred Image')
plt.show()


# # Create the pencil sketch image by dividing the grayscale image by inverted blurred image

# In[38]:


pencil_sketch = cv2.divide(gray_image, inverted_blurred_image, scale = 256.0)


# In[39]:


plt.imshow(pencil_sketch)
plt.axis('off')
plt.show()


# # Convert the pencil sketch to RGB

# In[42]:


pencil_sketch_rgb =  cv2.cvtColor(pencil_sketch,cv2.COLOR_GRAY2RGB)


# In[44]:


plt.imshow(pencil_sketch_rgb)
plt.title('Pencil Sketch')
plt.axis('off')
plt.show()


# In[ ]:




