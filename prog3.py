import numpy as np
from sklearn.decomposition import PCA
from PIL import Image

#Load the image
image = Image.open('C:\\Users\\VEJASHETTY\\mml\\WhatsApp Image 2024-03-13 at 
21.42.50_115e02f5.jpg')

image = image.convert('L') #greyscale
data = np.array(image)

#Apply PCA
pca = PCA(128) #Compression level adjusting
compressed_data = pca.fit_transform(data)
reconstructed_data = pca.inverse_transform(compressed_data)

#Save
compressed_image = Image.fromarray(reconstructed_data.astype(np.uint8))
compressed_image.save('C:\\Users\\VEJASHETTY\\mml\\WhatsApp Image 2024-03-13 at 
21.42.50_115e02f5.jpg')
