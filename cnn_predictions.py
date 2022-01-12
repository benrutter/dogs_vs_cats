from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import pandas as pd

label_generator = ImageDataGenerator(rescale=1.0/255)
label_iterator = label_generator.flow_from_directory(
    'data/to_predict',
    target_size=(128, 128),
    batch_size=32,
    shuffle=False,
    class_mode=None,
)

model = load_model('cnn', compile=True)

labels = model.predict(label_iterator)
labels = labels.clip(min=0.02, max=0.98)

subm = pd.read_csv('sample_submission.csv')
ids = [int(x.split("\\")[1].split(".")[0]) for x in label_iterator.filenames]

for i in range(len(ids)):
    subm.loc[subm.id == ids[i], "label"] = labels[:, 0][i]

subm.to_csv('submission.csv', index=False)
