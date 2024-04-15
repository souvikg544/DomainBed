from emotions import detect_emotion, init
import cv2
import numpy
import os
import glob
import shutil
from tqdm import tqdm

races = os.listdir('/ssd_scratch/cvit/souvik/race_per_7000')
batch_size = 128
init('cuda')


def read_images(images):
    image_list = []
    #print("--------------- Reading images ----------------")
    for img in tqdm(images):
        image_list.append(cv2.imread(img,cv2.IMREAD_COLOR))
    return image_list

def save(emotions,image_list,race): 
    i =0
    for img in image_list:        
        src = img
        dst = os.path.join('/ssd_scratch/cvit/souvik/balancedface/',race,emotions[i])
        shutil.copy2(src, dst)
        i+=1


for race in tqdm(races):
    path = os.path.join('/ssd_scratch/cvit/souvik/race_per_7000',race,'**/*.jpg')
    images = glob.glob(path,recursive = True)
    #image_list = read_images(images)
    
    for i in tqdm(range(0,len(images),batch_size)):
        try:
            image_list = read_images(images[i:i+batch_size])
            emotions = detect_emotion(image_list)
            save(emotions,images[i:i+batch_size],race)
        except Exception as e:
            print(e)
    print(f"Done for Domain = {race}")
    