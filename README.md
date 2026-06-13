# OCT Eye Disease Classifier

Retinal OCT image classification into four disease categories using 
transfer learning with DenseNet121.

## Classes
CNV, DME, DRUSEN, NORMAL

## Architecture
DenseNet121 pretrained on ImageNet (frozen base) with a custom 
classification head: GlobalAveragePooling → Dense(512, ReLU) → 
Dropout(0.5) → Dense(256, ReLU) → Dropout(0.5) → Dense(64, ReLU) → 
Softmax(4).

## Training
- Optimizer: Adam (lr=1e-4), Loss: Categorical Crossentropy  
- Callbacks: EarlyStopping (patience=5), ReduceLROnPlateau, 
  ModelCheckpoint  
- Augmentation: rotation, shift, zoom, shear, horizontal flip

## Results
- Training Accuracy: 83.2%  
- Validation Accuracy: 87.5%

## Stack
Python, TensorFlow, Keras, DenseNet121, OpenCV, Kaggle API
