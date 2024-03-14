if __name__=="__main__":
    import os
    import numpy as np
    from PIL import Image
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    import multiprocessing

    #replace with your place of the VinDR dataset (https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data)
    BASE_PATH = "/home/patrick/datasets/vinDR-CXR/"

    #folder for the input data
    INP_FOLDER = BASE_PATH + "train/"

    #folder for the output
    OUT_FOLDER = BASE_PATH + "train_png/"


    os.makedirs(OUT_FOLDER,exist_ok=True)

    def readDicom(path, voi_lut = True, fix_monochrome = True):
        #function from: https://www.kaggle.com/code/raddar/convert-dicom-to-np-array-the-correct-way
        dicom = pydicom.read_file(path)
        
        # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
        if voi_lut:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array
                
        # depending on this value, x-ray may look inverted - fix that:
        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data
            
        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
            
        return data

    def centerCrop(img):
        width, height = img.size
        newSize = min(width,height)

        left = (width - newSize)//2
        top = (height - newSize)//2
        right = (width + newSize)//2
        bottom = (height + newSize)//2

        img = img.crop((left, top, right, bottom))
        return img

    def convertImg(imgTup):
        inpFolder,outFolder,imgName = imgTup
        inpFile = inpFolder + imgName
        outFile = outFolder + imgName.split(".")[0] + ".png"

        img = readDicom(inpFile)
        img = Image.fromarray(img)
        img = centerCrop(img)

        #scale img to same size
        img = img.resize((1024,1024),Image.BILINEAR)

        img.save(outFile)
        #print("saved",outFile)

    imgList = [(INP_FOLDER,OUT_FOLDER, x) for x in os.listdir(INP_FOLDER)]
    
    print("start conversion; used processes:",multiprocessing.cpu_count(),"; number of files:",len(imgList))
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.map(convertImg,imgList)
    print("conversion completed")