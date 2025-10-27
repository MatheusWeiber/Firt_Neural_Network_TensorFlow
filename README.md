# Firt_Neural_Network_TensorFlow

Projeto: Classificação de Cobertura do Solo com Redes Neurais (TensorFlow/Keras)

🇧🇷 Versão em Português

Visão Geral

Este projeto é minha primeira implementação de uma rede neural profunda (Deep Learning) utilizando as bibliotecas TensorFlow e Keras. O objetivo foi construir, treinar e avaliar um modelo de Multi-Layer Perceptron (MLP) capaz de resolver um problema complexo de classificação multiclasse: o dataset Covertype da UCI.

O modelo alcançou uma acurácia de ~88% no conjunto de teste, demonstrando a capacidade do Keras em modelar relações não-lineares complexas.

O Desafio: Dataset Covertype

Problema: Classificação (Supervisionada).

Dataset: Contém 581.012 amostras de áreas florestais de 30x30m.

Features (Entrada): 54 colunas, incluindo dados quantitativos (Elevação, Inclinação, Distâncias) e dados binários (Tipos de Solo, Áreas Selvagens).

Alvo (Saída): 7 classes únicas (tipos de cobertura florestal, ex: "Spruce-Fir", "Lodgepole Pine").

Metodologia e Arquitetura

O pipeline do projeto seguiu as melhores práticas de Deep Learning:

Pré-processamento (sklearn):

Os dados foram divididos em conjuntos de treino (80%) e teste (20%).

As 10 features quantitativas foram normalizadas usando StandardScaler para otimizar a convergência da rede neural.

As 44 features binárias foram mantidas intactas.

Ajuste do Alvo: As 7 classes (originais 1-7) foram ajustadas para o formato esperado pelo Keras (0-6).

Construção do Modelo (keras.Sequential):

Camada de Entrada: Dense(100, activation='relu', input_shape=(54,))

Camada Oculta: Dense(50, activation='relu')

Regularização: Dropout(0.2) para prevenir overfitting (decoreba).

Camada de Saída: Dense(7, activation='softmax') para produzir as probabilidades das 7 classes.

Compilação e Treinamento:

Otimizador: adam

Função de Perda: sparse_categorical_crossentropy (ideal para alvos inteiros multiclasse).

Treino: O modelo foi treinado por 50 épocas, com batch_size=64, utilizando o conjunto de teste como validation_data para monitorar o desempenho.

Resultados

Acurácia Final: ~88%

Análise: O modelo demonstrou excelente capacidade de generalização. Os gráficos de histórico de treino (acurácia/loss vs. val_acurácia/val_loss) mostraram que o Dropout foi eficaz em prevenir overfitting. A matriz de confusão revelou que, embora o modelo seja forte, sua principal dificuldade é distinguir as Classes 1 e 2, que são geologicamente muito similares.

🇺🇸 English Version

Project: Forest Cover Type Classification with Neural Networks (TensorFlow/Keras)

Overview

This project is my first implementation of a deep neural network using the TensorFlow and Keras libraries. The objective was to build, train, and evaluate a Multi-Layer Perceptron (MLP) model capable of solving a complex, multi-class classification problem: the UCI Covertype dataset.

The model achieved an accuracy of ~88% on the test set, demonstrating Keras's capability in modeling complex non-linear relationships.

The Challenge: Covertype Dataset

Problem: Classification (Supervised).

Dataset: Contains 581,012 samples of 30x30m forest areas.

Features (Input): 54 columns, including quantitative data (Elevation, Slope, Distances) and binary data (Soil Types, Wilderness Areas).

Target (Output): 7 unique classes (types of forest cover, e.g., "Spruce-Fir", "Lodgepole Pine").

Methodology and Architecture

The project pipeline followed deep learning best practices:

Preprocessing (sklearn):

Data was split into training (80%) and testing (20%) sets.

The 10 quantitative features were normalized using StandardScaler to optimize the neural network's convergence.

The 44 binary features were passed through unchanged.

Target Adjustment: The 7 classes (originally 1-7) were adjusted to the Keras-expected format (0-6).

Model Building (keras.Sequential):

Input Layer: Dense(100, activation='relu', input_shape=(54,))

Hidden Layer: Dense(50, activation='relu')

Regularization: Dropout(0.2) to prevent overfitting.

Output Layer: Dense(7, activation='softmax') to produce the probability distribution for the 7 classes.

Compilation and Training:

Optimizer: adam

Loss Function: sparse_categorical_crossentropy (ideal for multi-class integer targets).

Training: The model was trained for 50 epochs with a batch_size=64, using the test set as validation_data to monitor performance.

Results

Final Accuracy: ~88%

Analysis: The model demonstrated excellent generalization capabilities. The training history plots (accuracy/loss vs. val_accuracy/val_loss) showed that Dropout was effective in preventing overfitting. The confusion matrix revealed that while the model is strong, its primary difficulty is in distinguishing between Classes 1 and 2, which are geologically very similar.
