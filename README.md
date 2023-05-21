# Transfer-Learning-MASK-R-CNN
This repository demonstrates my expertise in computer vision through a project on object detection using transfer learning with Mask R-CNN. By training the model on a custom dataset of people, it can accurately detect and mark people in images and videos, even beyond the dataset.
The project follows a comprehensive pipeline to achieve its objectives. It starts with the selection of an object detection dataset, which can be chosen from the Roboflow Public Datasets for convenience. The dataset is then split into training, validation, and testing subsets to ensure a robust evaluation of the model's performance.

The neural network training pipeline is built with careful consideration. Some layers of the Mask R-CNN model are frozen to leverage the pre-trained weights effectively. Data augmentation techniques, such as Albumentations, are applied to enhance the model's ability to generalize to different scenarios. Throughout the training process, the project plots and tracks the training and validation loss and accuracy scores, providing insights into the model's progression. Optionally, TensorBoard can be utilized for visualizing these metrics.

The model's performance is evaluated on the test set, measuring accuracy using metrics like mean Average Precision (mAP). This evaluation provides an objective assessment of the model's ability to accurately detect and classify people.

To demonstrate the model's applicability beyond static images, the project includes an inference step on a short video. A video titled "City Walk" from Tel Aviv, which is not part of the original dataset, is used to showcase the model's capabilities. The results of the video inference are saved, containing bounding boxes and class labels that mark the detected people.

In the final PDF explanation, I provide a detailed account of the implemented steps, including dataset selection, transfer learning techniques, data augmentation strategies, training pipeline, model evaluation, and video inference. This repository serves as a testament to my coding proficiency and expertise in computer vision tasks, providing recruiters with an opportunity to assess my skills in this domain.

Please note that the trained model weights and inference outputs may not be included in this repository.
