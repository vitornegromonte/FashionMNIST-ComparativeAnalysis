# FashionMNIST - Comparative Analysis
> Projeto prático da disciplina de Introdução à Aprendizagem Profunda (IF867)

## Treine e avalie 4 modelos de classificação para a base de dados do FashionMNIST:
- [Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
- [PyTorch](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html)

## Atividade:
1. Um modelo base que não seja uma rede neural, como decision tree, xgboost, random forest, etc. Recomendação: use o sklearn (https://scikit-learn.org/).
2. Uma MLP
3. Uma rede convolucional criada por ti. Recomendação: https://pytorch.org/
4. Use um modelo pré treinado já consolidado na literatura para fazer transfer learning. Recomendações: https://pytorch.org/hub/pytorch_vision_vgg/

## Compare os resultados dos modelos:
- Plote gráficos que mostrem as acurácias de cada modelo
- Indique qual foi a classe na qual o modelo teve pior performance (indique qual métrica usou para concluir isso e faça para cada modelo)
- Argumente qual o melhor modelo levando em consideração o tempo de execução e acurácia.

> Recomendação use: https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html .

## Recomendações gerais:
- Faça um template de treino, validação e teste que funcione para uma API de modelo.
- Crie a API para cada modelo que será usado e use o template.


# Resultados:
|    Model             |  Accuracy  | Execution Time(s) |
|----------------------|------------|--------------------|
| Random Forest        |   0.8773   |      98.2990       |
| SVCLinear            |   0.8463   |     362.3468       |
| SVM                  |   0.9002   |     258.6090       |
| Decision tree        |   0.8008   |      21.8467       |
| KNN                  |   0.8577   |       0.0227       |
| Logistic Regression  |   0.8413   |      26.3970       |
| Naive Bayes          |   0.5856   |       0.2897       |
| AdaBoost             |   0.5928   |     240.3844       |
| VGG16                |   0.7850   |       8.5534       |
| VGG19                |   0.8290   |       9.5304       |
| ResNet50             |   0.8250   |      12.0556       |
| ResNet152            |   0.3850   |      22.9346       |
| InceptionV3          |   0.6310   |       9.7400       |
| DenseNet121          |   0.6790   |      13.7981       |
| DenseNet201          |   0.6580   |     121.0056       |
| MLP                  |   0.9111   |      27.8681       |
| CNN_model            |   0.9111   |      27.8681       |
