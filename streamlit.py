import streamlit as st
import dataset
import model
import hyper_parameters
import train
import preprocessing


st.title("Model Training Configuration")


st.header("Model Configuration")

model_name = st.selectbox("Select the Model", ["","BERT", "vision_transformer"])


type_of_fine_tuning = st.selectbox("Select the type of fine method", ["","LoRA", "QLoRA","Without_Peft"])



if model_name == "BERT":
    st.subheader("BERT Model Parameters")
    freeze_weights = st.checkbox("Freeze Model Weights")
    num_train_epochs = st.number_input("Number of Training Epochs", value=3, min_value=1)
    per_device_train_batch_size = st.number_input("Batch Size for Training", value=8, min_value=1)
    per_device_eval_batch_size = st.number_input("Batch Size for Evaluation", value=8, min_value=1)
    if type_of_fine_tuning == "LoRA":
        lora_alpha = st.number_input("LoRA Alpha", value=64, min_value=1)
        lora_dropout = st.number_input("LoRA Dropout", value=0.16)
        r = st.number_input("R", value=0.1)
    
    if type_of_fine_tuning == "QLoRA":
        pass

    if type_of_fine_tuning == "Without_Peft":
        pass

    train = st.button("Start Training")

    if train:
        pass

if model_name == "vision_transformer":

    model_checkpoint = "google/vit-base-patch16-224-in21k"
    data = dataset.DatasetFactory.get_dataset(model)
    
    st.subheader("Vision Transformers Model Parameters")
    freeze_weights = st.checkbox("Freeze Model Weights")
    num_train_epochs = st.number_input("Number of Training Epochs", value=3, min_value=1)
    per_device_train_batch_size = st.number_input("Batch Size for Training", value=8, min_value=1)
    per_device_eval_batch_size = st.number_input("Batch Size for Evaluation", value=8, min_value=1)
    if type_of_fine_tuning == "LoRA":
        lora_alpha = st.number_input("LoRA Alpha", value=64, min_value=1)
        lora_dropout = st.number_input("LoRA Dropout", value=0.16)
        r = st.number_input("R", value=0.1)
    
    if type_of_fine_tuning == "QLoRA":
        pass

    if type_of_fine_tuning == "Without_Peft":
        pass

    train = st.button("Start Training")

    if train:
        pass