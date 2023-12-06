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
