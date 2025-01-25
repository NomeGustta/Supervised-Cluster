import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO)

def load_and_preprocess_data():
    """
    Função para carregar e pré-processar os dados do conjunto Iris.
    - Carrega os dados
    - Escalona as features
    """
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Escalonamento das features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    df = pd.DataFrame(X_scaled, columns=iris.feature_names)
    df['species'] = y
    logging.info("Dados carregados e escalonados com sucesso.")
    return X_scaled, y, iris

def train_model(X_train, y_train):
    """
    Função para treinar o modelo de Regressão Logística com ajuste de hiperparâmetros.
    """
    # Definindo um Pipeline com o escalonamento e o modelo
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('model', LogisticRegression(max_iter=200))
    ])
    
    # Definindo o grid de parâmetros para o ajuste
    param_grid = {
        'model__C': [0.1, 1, 10],
        'model__solver': ['lbfgs', 'liblinear']
    }

    # Usando GridSearchCV para ajuste de hiperparâmetros
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    logging.info(f"Melhor hiperparâmetro encontrado: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, iris):
    """
    Função para avaliar o modelo treinado usando as métricas de desempenho.
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Acurácia: {accuracy:.2f}")
    
    logging.info("Relatório de Classificação:")
    logging.info(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    logging.info("Matriz de Confusão:")
    logging.info(cm)
    
    # Exibindo a matriz de confusão de forma visual
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False,
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel("Predição")
    plt.ylabel("Verdadeiro")
    plt.title("Matriz de Confusão - Regressão Logística")
    plt.show()

def visualize_pca(X_test, y_test, y_pred, iris):
    """
    Função para visualizar as previsões e classes reais em 3D após PCA.
    """
    pca_3d = PCA(n_components=3)
    X_test_pca_3d = pca_3d.fit_transform(X_test)

    # Mostra a variância explicada pelas componentes principais
    explained_variance = pca_3d.explained_variance_ratio_
    logging.info(f"Variância explicada: PC1: {explained_variance[0]*100:.2f}%, PC2: {explained_variance[1]*100:.2f}%, PC3: {explained_variance[2]*100:.2f}%")

    colors = ['red', 'blue', 'green']
    
    # Visualização das classes REAIS em 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, color, label in zip([0, 1, 2], colors, iris.target_names):
        ax.scatter(
            X_test_pca_3d[y_test == i, 0],
            X_test_pca_3d[y_test == i, 1],
            X_test_pca_3d[y_test == i, 2],
            c=color,
            label=label,
            s=50,
            alpha=0.8
        )
    
    ax.set_title('Visualização 3D (Classes Reais)')
    ax.set_xlabel('Componente 1')
    ax.set_ylabel('Componente 2')
    ax.set_zlabel('Componente 3')
    ax.legend()
    plt.show()

    # Visualização das classes PREDITAS em 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for i, color, label in zip([0, 1, 2], colors, iris.target_names):
        ax.scatter(
            X_test_pca_3d[y_pred == i, 0],
            X_test_pca_3d[y_pred == i, 1],
            X_test_pca_3d[y_pred == i, 2],
            c=color,
            label=label,
            s=50,
            alpha=0.8
        )

    ax.set_title('Visualização 3D (Classes Preditas)')
    ax.set_xlabel('Componente 1')
    ax.set_ylabel('Componente 2')
    ax.set_zlabel('Componente 3')
    ax.legend()
    plt.show()

def main():
    """
    Função principal para orquestrar o fluxo do código.
    """
    # Carregar e pré-processar os dados
    X, y, iris = load_and_preprocess_data()

    # Dividir os dados em treino e teste (70%-30%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Treinar o modelo
    model = train_model(X_train, y_train)

    # Avaliar o modelo
    evaluate_model(model, X_test, y_test, iris)  # Passando iris como argumento

    # Visualizar o PCA
    visualize_pca(X_test, y_test, model.predict(X_test), iris)

if __name__ == "__main__":
    main()
