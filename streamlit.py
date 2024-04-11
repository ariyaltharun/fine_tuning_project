import streamlit as st
import dataset
import model
import hyper_parameters
import train
import preprocessing
import peft_techniques

st.title("Model Training Configuration")

st.header("Model Configuration")

model_name = st.selectbox("Select the Model", ["","BERT", "vision_transformer"])

type_of_fine_tuning = st.selectbox("Select the type of fine-tuning method", ["","LoRA", "QLoRA","Without_Peft"])

if model_name == "BERT":
    pass

if model_name == "vision_transformer":
    st.subheader("Vision Transformers Model Parameters")
    freeze_weights = st.checkbox("Freeze Model Weights")
    num_train_epochs = st.number_input("Number of Training Epochs", value=3, min_value=1)
    per_device_train_batch_size = st.number_input("Batch Size for Training", value=8, min_value=1)
    learning_rate = st.number_input("Learning Rate", value=0.001, min_value=0.0001, step=0.001)

    if type_of_fine_tuning == "LoRA":
        lora_alpha = st.number_input("LoRA Alpha", value=64, min_value=1)
        lora_dropout = st.number_input("LoRA Dropout", value=0.16)
        r = st.number_input("R", value=16, min_value=1)

        train_button = st.button("Start Training")
        
        if train_button:
            with st.spinner("Training in progress..."):
                data = dataset.DatasetFactory.get_dataset(model_name)
                parameter = {
                    "r": r,
                    "lora_alpha": lora_alpha,
                    "target_modules": ["query", "value"], 
                    "lora_dropout": lora_dropout,
                    "bias": "none",
                }
                gen_params = {
                    "batch_size": per_device_train_batch_size,
                    "learning_rate": learning_rate, 
                    "num_train_epochs": num_train_epochs,
                }

                peft_params = hyper_parameters.HyperParameterFactory.get_peft_parameters("lora", parameter)
                general_params = hyper_parameters.HyperParameterFactory.get_general_parameters(gen_params)

                vision_model = model.ModelFactory.create(model_name, data)
                processor = preprocessing.PreprocessingFactory("google/vit-base-patch16-224-in21k")
                train_ds, val_ds = processor.get_image_preprocessor(data)

                lora_model = peft_techniques.PeftFactory.create_peft_method(vision_model, peft_params)

                args = hyper_parameters.HyperParameterFactory.get_general_parameters(general_params)

                trainer = train.TrainFactory.get_trainer(vision_model, args, train_ds, val_ds, processor)
                train_results = trainer.train()
                validation_results = trainer.evaluate(val_ds)

                st.success("Training completed!")
                st.write("Training Results:", train_results)
                st.write("Validation Results:", validation_results)
