import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import preprocess_input

def get_class_names(data_dir='Weather_Dataset'):
    """
    Get class names from the dataset directory.
    """
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=3,
        image_size=(256, 256),
        batch_size=32
    )
    return train_ds.class_names

def create_model():
    """
    Create model architecture identical to training.
    """
    resnet_model = Sequential()
    
    pretrained_model = ResNet50(
        include_top=False,
        input_shape=(256, 256, 3),
        pooling='avg',
        weights='imagenet'
    )
    # Freeze all layers
    for layer in pretrained_model.layers:
        layer.trainable = False
    
    resnet_model.add(pretrained_model)
    resnet_model.add(Flatten())
    resnet_model.add(Dense(512, activation='relu'))
    resnet_model.add(Dense(3, activation='softmax'))
    
    resnet_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return resnet_model

def process_video(video_path, model, class_names, output_path=None):
    """
    Process the video with predictions and optionally save output.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    out = None
    if output_path:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get the original FPS
        if fps == 0:
            fps = 30  # Fallback if FPS not available
        
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocessing
        processed_frame = cv2.resize(frame, (256, 256))
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        processed_frame = preprocess_input(processed_frame)  # Optionally use ResNet-specific preprocessing
        processed_frame = np.expand_dims(processed_frame, axis=0)
        
        # Prediction
        prediction = model.predict(processed_frame, verbose=0)
        predicted_class = class_names[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])
        
        # Overlay prediction on frame
        text = f'Weather: {predicted_class} ({confidence:.2f})'
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('Weather Classification', frame)
        if out is not None:
            out.write(frame)
        
        # Break on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    data_dir = 'Weather_Dataset'
    class_names = get_class_names(data_dir)
    print("Found classes:", class_names)
    
    model = create_model()
    
    weights_path = "./Weights/model.h5"  # Replace with your valid weights file
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print(f"Successfully loaded weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {str(e)}")
            exit(1)
    else:
        print(f"Weight file not found: {weights_path}")
        exit(1)
    
    # Replace with your video path
    video_path = r"C:\Users\Administrator\Downloads\Rain\istockphoto-1439874930-640_adpp_is.mp4"
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        exit(1)
    
    output_path = 'weather_prediction.mp4'
    process_video(video_path, model, class_names, output_path)
