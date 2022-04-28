from PIL import Image
from PIL import ImageDraw

img = Image.open('2.jpg')
x, y, w, h = 21.99091,  59.31325, 564.24445, 395.77777
width, height = img.size
print(width, height)
w1, h1 = 640, 480
#img2 = img.resize((250, 650))
dra = ImageDraw.ImageDraw(img)
x1 = x/w1 * width
y1 = y/h1 * height
x2 = w/w1 * width
y2 = h/h1 * height

dra.rectangle(((x1, y1), (x2, y2)), fill=None, outline='red', width=5)
img.show()
