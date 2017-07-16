__author__ = 'bziobrow'


import imageio
import numpy as np

imageio.plugins.ffmpeg.download()

filename = 'video_sources/1.mp4'
vid = imageio.get_reader(filename,'ffmpeg')
nums = []
nums = np.arange(0,287959,100)
for num in nums:
    image = vid.get_data(num)
    image.shape
    #export image in grayscale
    #imageio.imwrite('video_output/{}.jpg'.format(num),image[:,:,0],format='jpg')
    imageio.imwrite('video_output/{}.jpg'.format(num),image[:,:,:],format='jpg')
    #imageio.imwrite('imageio:grayscale.jpg',image[:,:,0])