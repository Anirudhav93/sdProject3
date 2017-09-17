from keras.models import Sequential
from keras.layers.core import Dense, Flatten ,Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np       
import cv2                 
import csv

     

def Preprocess(img):
    '''
    Input :- opencv image
    Output:- opencv_image
    Gaussian Blur, Cropping and Color Space tranformation
    '''

    new_img = img[50:140,:,:]

    new_img = cv2.GaussianBlur(new_img, (3,3), 0)

    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)

    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img

def DistortionNoise(img, angle):
    ''' 
    Input:- Image and Angle
    Output:- Image and Angle
    Random noises and distortions to the image
    '''
    new_img = img.astype(float)
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (new_img[:,:,0] + value) > 255 
    if value <= 0:
        mask = (new_img[:,:,0] + value) < 0
    new_img[:,:,0] += np.where(mask, 0, value)
    h,w = new_img.shape[0:2]
    mid = np.random.randint(0,w)
    factor = np.random.uniform(0.6,0.8)
    if np.random.rand() > .5:
        new_img[:,0:mid,0] *= factor
    else:
        new_img[:,mid:w,0] *= factor
    horizon = 2*h/5
    v_shift = np.random.randint(-h/8,h/8)
    pts1 = np.float32([[0,horizon],[w,horizon],[0,h],[w,h]])
    pts2 = np.float32([[0,horizon+v_shift],[w,horizon+v_shift],[0,h],[w,h]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    new_img = cv2.warpPerspective(new_img,M,(w,h), borderMode=cv2.BORDER_REPLICATE)
    return (new_img.astype(np.uint8), angle)


def generate(image_paths, angles, batch_size=128, validation_flag=False):
    image_paths, angles = shuffle(image_paths, angles)
    X,y = ([],[])
    while True:       
        for i in range(len(angles)):
            img = cv2.imread(image_paths[i])
            angle = angles[i]
            img = Preprocess(img)
            if not validation_flag:
                img, angle = DistortionNoise(img, angle)
            X.append(img)
            y.append(angle)
            if len(X) == batch_size:
                yield (np.array(X), np.array(y))
                X, y = ([],[])
                image_paths, angles = shuffle(image_paths, angles)

            img = cv2.flip(img, 1)
            angle *= -1
            X.append(img)
            y.append(angle)
            if len(X) == batch_size:
                yield (np.array(X), np.array(y))
                X, y = ([],[])
                image_paths, angles = shuffle(image_paths, angles)



'''
Main program 
'''
if __name__ == "__main__":

    using_my_data = True    
    data_to_use = [using_my_data] 
    csv_path = 'driving_log.csv' 
    
    image_paths = []
    angles = []

    # Import driving data from csv
    with open(csv_path, newline='') as f:
        driving_data = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
    
    
        # Gather data 
        for row in driving_data[1:]:
            if float(row[6]) < 0.1 :
                continue
            # center image
            image_paths.append(row[0])
            angles.append(float(row[3]))
           
            
            # left image
            image_paths.append(row[1])
            angles.append(float(row[3])+0.25)
            
            
            # right image
            image_paths.append(row[2])
            angles.append(float(row[3])-0.25)
    
    image_paths = np.array(image_paths)
    angles = np.array(angles)
    
    
    #flatten the distribution of the dataset threshold is half the average
    num_bins = 23
    avg_samples_per_bin = len(angles)/num_bins
    hist, bins = np.histogram(angles, num_bins)
    keep_probs = []
    target = avg_samples_per_bin * .5
    for i in range(num_bins):
        if hist[i] < target:
            keep_probs.append(1.)
        else:
            keep_probs.append(1./(hist[i]/target))
    remove_list = []
    for i in range(len(angles)):
        for j in range(num_bins):
            if angles[i] > bins[j] and angles[i] <= bins[j+1]:
                # delete from X and y with probability 1 - keep_probs[j]
                if np.random.rand() > keep_probs[j]:
                    remove_list.append(i)
    image_paths = np.delete(image_paths, remove_list, axis=0)
    angles = np.delete(angles, remove_list)

    
    # split into train/test sets
    image_paths_train, image_paths_test, angles_train, angles_test = train_test_split(image_paths, angles,
                                                                                      test_size=0.05, random_state=42)
    

    model = Sequential()
    
        # Normalize
    model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(66,200,3)))
    
        # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    
        #model.add(Dropout(0.50))
        
        # Add two 3x3 convolution layers (output depth 64, and 64)
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    
        # Add a flatten layer
    model.add(Flatten())
    
        # Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(ELU())
        #model.add(Dropout(0.50))
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(ELU())
        #model.add(Dropout(0.50))
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(ELU())
        #model.add(Dropout(0.50))
    
        # Add a fully connected output layer
    model.add(Dense(1))
    
        # Compile and train the model, 
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    

    
        # initialize generators
    trainingData = generate(image_paths_train, angles_train, validation_flag=False, batch_size=64)
    validationData = generate(image_paths_train, angles_train, validation_flag=True, batch_size=64)

    
    model.fit_generator(trainingData, validation_data=validationData, nb_val_samples=2560, samples_per_epoch=23040, 
                                      nb_epoch=5) 
    
    print(model.summary())
    

        # Save model data
    model.save('./model.h5')

