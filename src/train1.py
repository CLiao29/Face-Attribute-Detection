import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2  import MobileNetV2
from tensorflow.keras.layers import Dense , BatchNormalization , Dropout
from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint


class celeb_dataframe():
    '''Wraps the celebA dataset, allowing an easy way to:
       - Select the features of interest,
       - Split the dataset into 'training', 'test' or 'validation' partition.
    '''
    def __init__(self,main_folder = "data\celeba" , selected_features = [] , drop_features = []):
        self.main_folder = main_folder
        self.images_folder   = os.path.join(main_folder, 'img_align_celeba')
        self.attr_path = os.path.join(main_folder , "list_attr_celeba.csv")
        self.features_name = []
        self.prepare(drop_features , selected_features)
        
    def prepare(self,drop_features, selected_features ):
        
        #attributes selection
        if len(selected_features) == 0  :
            self.attributes = pd.read_csv(self.attr_path)
            self.num_features = 40
        else:
            self.num_features = len(selected_features)
            selected_features = selected_features.copy()
            selected_features.append("image_id")
            self.attributes = pd.read_csv(self.attr_path)[selected_features]
        
        #removing features
        if len(drop_features) != 0:
            for feature in drop_features:
                if feature in self.attributes:
                    self.attributes = self.attributes.drop(feature , axis = 1)
                    self.num_features -= 1
            
        self.attributes.set_index("image_id" , inplace = True)
        self.attributes.replace(to_replace = -1 , value = 0 , inplace = True)
        self.attributes["image_id"] = list(self.attributes.index)
        self.features_name = list(self.attributes.columns)[:-1]
        
        return self.attributes

def main():
    batch_size = 64
    split = 0.2
    epochs = 5
    celeb_df = celeb_dataframe()
    features  = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
       'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
       'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
       'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
       'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
       'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
       'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
       'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
       'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    celeb = celeb_df.prepare(drop_features=[],selected_features= features) #taking all the features for training.
    split_ind = int((1 - split) * celeb.shape[0])
    train_df = celeb[:split_ind]
    validation_df = celeb[split_ind:]

    # augumentations for training set:
    train_datagen = ImageDataGenerator(rotation_range=20, 
                                   rescale=1./255, 
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True, 
                                   fill_mode='nearest')

    valid_datagen = ImageDataGenerator(rescale= 1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=celeb_df.images_folder,
        x_col='image_id',
        y_col=celeb_df.features_name,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='raw',
        shuffle = True
    )

    validation_generator = valid_datagen.flow_from_dataframe(dataframe=validation_df,
                                                         directory=celeb_df.images_folder,
                                                         x_col='image_id',
                                                         y_col=celeb_df.features_name,
                                                         target_size=(224, 224),
                                                         batch_size=batch_size,
                                                         class_mode='raw'
                                                        )

    num_features = len(celeb_df.features_name)
    cls = classifier(num_features)
    cls.compile(loss='binary_crossentropy',
              optimizer= tf.keras.optimizers.Adam(0.001),
              metrics='binary_accuracy')

    earlystop = EarlyStopping(monitor="val_binary_accuracy", patience= 3)
    checkpoint_filepath = "..\data" + f"/weights-FC{celeb_df.num_features}-MobileNetV2" + "{val_binary_accuracy:.2f}.hdf5"
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_binary_accuracy',
        mode='max',
        save_best_only=True)
    
    history = cls.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=len(train_generator),
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks = [earlystop , model_checkpoint],
        max_queue_size=1,
        verbose=1)
    
    path = "model\My_celebA_attr_Classifier_model"
    cls.save(path)
    cls.save_weights('model\weights')

    

    
def classifier(num_features):
    base = MobileNetV2(input_shape = (224,224,3),
                      weights = None,
                      include_top=False,
                      pooling = "avg")
    
    x = base.output
    x = Dense(1536, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    top = Dense(num_features, activation='sigmoid')(x)
    classifier = tf.keras.models.Model(base.input,top)
    
    return classifier


if __name__ == '__main__':
    main()