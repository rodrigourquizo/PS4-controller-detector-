import os 
from PIL import Image

SIZE = 300
images = "./data"
resized_images = "./images"
train_images = os.path.join(images, "train")
test_images = os.path.join(images, "test")
train_resized_images = os.path.join(resized_images , "train")
test_resized_images = os.path.join(resized_images , "test")

def resize_images(from_, save_in, size):
    for i in os.listdir(from_):
        img = Image.open(os.path.join(from_, i))
        img = img.resize((size,size), Image.ANTIALIAS)
        img.save(os.path.join(save_in, i))
        
def delete(folder):
    for x in os.listdir(folder):
        os.remove(os.path.join(folder,x))
        
if __name__ == "__main__":
    delete(train_resized_images)
    delete(test_resized_images)
    resize_images(train_images, train_resized_images, SIZE)
    resize_images(test_images, test_resized_images, SIZE)

