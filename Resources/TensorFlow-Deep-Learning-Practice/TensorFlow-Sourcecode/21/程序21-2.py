from numpy import ogrid, repeat, newaxis
from skimage import io
import  numpy as np
size = 3
x, y = ogrid[:size, :size]
img = repeat((x + y)[..., newaxis], 3, 2) / 12.
io.imshow(img, interpolation='none')
io.show()
