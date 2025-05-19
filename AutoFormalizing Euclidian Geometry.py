import numpy as np
import pandas as pd
import UniGeo.Congruent
import UniGeo.Parallel
import UniGeo.Quadrilateral
import UniGeo.Similarity
import UniGeo.Triangle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os

class GeometryDataset:
    def __init__(self, dataset_path="E:\\Code & Dataset.zip\\Code & Dataset\\Dataset\\UniGeo"):
    self.dataset_path = dataset_path
    dataset_path = "Congruent.lean" 
    dataset_path = "Quadrilateral.lean" 
    dataset_path = "Parallel.lean" 
    dataset_path = "Relations.lean" 
    dataset_path = "Similarity.lean" 
    dataset_path = "Traingle.lean" 
    
        self.dataset_path = dataset_path
        self.data = None

    def load_data(self):
       
        try:
            self.data = pd.read_csv(self.dataset_path)
            print(f"Dataset loaded: {len(self.data)} examples.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

    def preprocess(self, few_shot=False, num_shots=1):

        if self.data is None:
            print("Dataset not loaded.")
            return None, None, None, None

        X = self.data["Statement"]
        y = self.data["Label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if few_shot:
            X_train, y_train = X_train[:num_shots], y_train[:num_shots]

        return X_train, X_test, y_train, y_test

class Geometry4Model:
    def __init__(self, dataset_path="E:\\Code & Dataset.zip\\Code & Dataset\\Dataset\\UniGeo"):
    self.dataset_path = dataset_path
    dataset_path = "Congruent.lean" 
    dataset_path = "Quadrilateral.lean" 
    dataset_path = "Parallel.lean" 
    dataset_path = "Relations.lean" 
    dataset_path = "Similarity.lean" 
    dataset_path = "Traingle.lean" 
    def __init__(self, model_name="bert-base-uncased"):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)  # 5 categories
        self.trainer = None

    def tokenize_data(self, X, y):
       
        return self.tokenizer(list(X), padding=True, truncation=True, max_length=512, return_tensors="pt"), torch.tensor(y.values)

    def train(self, X_train, y_train, X_test, y_test):
       
        train_encodings, train_labels = self.tokenize_data(X_train, y_train)
        test_encodings, test_labels = self.tokenize_data(X_test, y_test)
class Geometry4OMiniModel:
    def __init__(self, dataset_path="E:\\Code & Dataset.zip\\Code & Dataset\\Dataset\\UniGeo"):
    self.dataset_path = dataset_path
    dataset_path = "Congruent.lean" 
    dataset_path = "Quadrilateral.lean" 
    dataset_path = "Parallel.lean" 
    dataset_path = "Relations.lean" 
    dataset_path = "Similarity.lean" 
    dataset_path = "Traingle.lean" 
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
       
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

    def preprocess(self, statements, images):
        
        inputs = self.processor(
            text=list(statements),
            images=images,  # Assumes images are PIL.Image objects
            return_tensors="pt",
            padding=True,
        )
        return inputs

    def predict(self, statements, images):
        
        inputs = self.preprocess(statements, images)
        outputs = self.model(**inputs)
        logits_per_text = outputs.logits_per_text  
        predictions = torch.argmax(logits_per_text, dim=1)
        return predictions.numpy()


        class Geometry(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]

            def __len__(self):
                return len(self.labels)

        train_dataset = GeometryDataset(train_encodings, train_labels)
        test_dataset = GeometryDataset(test_encodings, test_labels)


        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="steps"
        )


        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer
        )


        self.trainer.train()

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model and returns predictions.
        """
        test_encodings, _ = self.tokenize_data(X_test, y_test)
        predictions = self.model(**test_encodings).logits
        return torch.argmax(predictions, axis=1).numpy()

class GeometryEvaluator:
    def __init__(self):
        self.categories = ["Triangle", "Similarity", "Congruent", "Quadrilateral", "Parallel"]

    def calculate_metrics(self, y_true, y_pred):
       
        results = {}
        for category in self.categories:
            indices = [i for i, label in enumerate(y_true) if label == category]
            if indices:
                accuracy = accuracy_score([y_true[i] for i in indices], [y_pred[i] for i in indices])
                results[category] = accuracy * 100
            else:
                results[category] = "N/A"  

        return results

    def display_results(self, results_1shot, results_5shot):
        
        data = {"Category": self.categories}
        data["1-shot Accuracy"] = [results_1shot.get(category, "N/A") for category in self.categories]
        data["5-shot Accuracy"] = [results_5shot.get(category, "N/A") for category in self.categories]

        df = pd.DataFrame(data)
        print("\nResults Table:")
        print(df.to_string(index=False))

class Geometry4Evaluator(GeometryEvaluator):
class Geometry4OMinivaluator(GeometryEvaluator):
    def evaluate_shots(self, dataset, images, num_shots):
       
        
        selected_statements = []
        selected_images = []
        selected_labels = []

        for category in self.categories:
            indices = dataset.data[dataset.data["Label"] == category].index
            selected_indices = indices[:num_shots] if len(indices) >= num_shots else indices
            selected_statements.extend(dataset.data.loc[selected_indices, "Statement"])
            selected_images.extend([images[idx] for idx in selected_indices])
            selected_labels.extend(dataset.data.loc[selected_indices, "Label"])

        
        y_pred = model_4V.predict(selected_statements, selected_images)

        return self.calculate_metrics(selected_labels, y_pred)


if __name__ == "__main__":
    # Step 1: Load dataset
    dataset_path = "Congruent.lean" 
    dataset_path = "Quadrilateral.lean" 
    dataset_path = "Parallel.lean" 
    dataset_path = "Relations.lean" 
    dataset_path = "Similarity.lean" 
    dataset_path = "Traingle.lean" 

    dataset = GeometryDataset(dataset_path)
    dataset.load_data()

    # 1-Shot Evaluation
    X_train, X_test, y_train, y_test = dataset.preprocess(few_shot=True, num_shots=1)
    model_1shot = GeometryModel()
    model_1shot.train(X_train, y_train, X_test, y_test)
    y_pred_1shot = model_1shot.evaluate(X_test, y_test)

    # 5-Shot Evaluation
    X_train, X_test, y_train, y_test = dataset.preprocess(few_shot=True, num_shots=5)
    model_5shot = GeometryModel()
    model_5shot.train(X_train, y_train, X_test, y_test)
    y_pred_5shot = model_5shot.evaluate(X_test, y_test)

    # Step 4: Display results
    evaluator = GeometryEvaluator()
    metrics_1shot = evaluator.calculate_metrics(y_test, y_pred_1shot)
    metrics_5shot = evaluator.calculate_metrics(y_test, y_pred_5shot)
    evaluator.display_results(metrics_1shot, metrics_5shot)

    if __name__ == "__main__":
    # Step 1: Load dataset
    dataset_path = "Congruent.lean" 
    dataset_path = "Quadrilateral.lean" 
    dataset_path = "Parallel.lean" 
    dataset_path = "Relations.lean" 
    dataset_path = "Similarity.lean" 
    dataset_path = "Traingle.lean" 
    dataset = GeometryDataset(dataset_path)
    dataset.load_data()
    
    model_4Omini = Geometry4OminiModel()
    evaluator_4Omini = Geometry4OminiEvaluator()
    print("\n")
    metrics_4_1shot = evaluator_4.evaluate_shots(dataset, images, num_shots=1)
    evaluator_4.display_results(metrics_4omini_1shot, {})

    
    print("\n")
    metrics_4_5shot = evaluator_4.evaluate_shots(dataset, images, num_shots=5)
    evaluator_4.display_results({}, metrics_4_5shot)

    print("\n")
    metrics_4Omini_1shot = evaluator_4Omini.evaluate_shots(dataset, images, num_shots=1)
    evaluator_4Omini.display_results(metrics_4omini_1shot, {})

    
    print("\n")
    metrics_4Omini_5shot = evaluator_4Omini.evaluate_shots(dataset, images, num_shots=5)
    evaluator_4Omini.display_results({}, metrics_4Omini_5shot)
