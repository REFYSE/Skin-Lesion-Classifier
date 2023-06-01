import os
import pandas as pd
from glob import glob

# A dictionary of the abbreviations to the lesion type for reference
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
classes = sorted(lesion_type_dict.keys())

def get_df_train(train_path):
    image_paths = glob(os.path.join(train_path, '*/*.jpg'))
    image_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in image_paths}
    df_train = pd.read_csv(os.path.join(train_path,'HAM10000_metadata.csv'))
    df_train['path'] = df_train['image_id'].map(image_dict.get)
    df_train['class'] = df_train['dx'].apply(lambda x: classes.index(x))
    # Not using patient metadata
    df_train = df_train.drop(['dx_type', 'age', 'sex', 'localization'], axis=1)
    return df_train

def get_df_test(test_path):
    df_test = pd.read_csv(os.path.join(test_path, 'ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv'))
    df_test = df_test.melt(id_vars=['image'], var_name='dx', value_name='belongs_to_class')\
                         .query('belongs_to_class != 0')\
                         .drop(columns=['belongs_to_class'])
    df_test['dx'] = df_test['dx'].str.lower()
    df_test['class'] = df_test['dx'].apply(lambda x: classes.index(x))
    df_test = df_test.rename(columns={'image': 'image_id'})
    df_test['path'] = df_test['image_id'].apply(lambda x: os.path.join(test_path, 'ISIC2018_Task3_Test_Input/ISIC2018_Task3_Test_Input/' + x + '.jpg'))
    df_test = df_test.reset_index(drop=True)
    return df_test    

def get_train_val_split(df_train, df_val):
    # Remove test data from train data
    df_train_unique = df_train[~df_train['lesion_id'].isin(df_val['lesion_id'])]
    df_train_unique.reset_index(inplace=True)
    df_val.reset_index(inplace=True)
    return df_train_unique, df_val