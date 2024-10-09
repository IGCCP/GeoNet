## Geological Object Recognition in Geological Maps Through Data Augmentation and Transfer Learning Techniques ##

This study proposes an innovative method to improve geological object recognition by leveraging legend data for data augmentation and using transfer learning with EfficientNet. The approach enhances model performance, particularly for texture-rich datasets, by increasing data diversity and reducing training time, leading to more accurate and efficient geological feature classification.

### folder ###
*/data/input* data after text removal

*/data/input* data after data overlays and data augmentation

*/data/ov.fit* Digital Elevation Model (DEM) data used in the manuscript

*/or_data* original (raw) data

*/model* saved model

### files ###
**EfficientNet.py** EfficientNet model

**data_augmentation.py** Data augmentation processing

**image_classification.py** A fully connected layer was added to the model to adapt it for geological feature classification

**image_classification_evaluate.py** Evaluation of model results
