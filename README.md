# 📝 **README - Projeto de Classificação com Regressão Logística e Visualização em 3D**

> **TL;DR:** Este projeto implementa um classificador de **Regressão Logística** para o conjunto de dados **Iris**. Após o pré-processamento e ajuste de hiperparâmetros via **`GridSearchCV`**, analisamos o desempenho por meio de métricas de classificação, como matriz de confusão. Além disso, o **PCA** é utilizado para gerar visualizações **3D** das classes reais e preditas.

---

## 📌 **Destaques do Projeto**
- **Pré-processamento Automático**: Escalonamento dos dados (`StandardScaler`).
- **Pipeline Simplificado**: Integra o escalonamento e o modelo em uma única estrutura.
- **Busca de Hiperparâmetros**: Usa `GridSearchCV` para encontrar a melhor configuração do modelo.
- **Métricas de Avaliação**: Acurácia, relatório de classificação e matriz de confusão em formato de *heatmap*.
- **Visualização 3D em PCA**: Mostra a separação das classes reais e preditas em três dimensões.

---

## ✨ **O que este código faz?**
1. **Carrega e Pré-processa o Conjunto de Dados Iris**  
   - Utiliza o `StandardScaler` para escalonar as variáveis numéricas.  
   - Organiza os dados em um DataFrame e adiciona a coluna de rótulos (espécies).

2. **Treina o Modelo de Regressão Logística**  
   - Define um `Pipeline` que inclui o escalonamento e a Regressão Logística.  
   - Ajusta hiperparâmetros via `GridSearchCV`, determinando a melhor combinação de parâmetros.

3. **Avalia o Modelo**  
   - Separa dados em treino e teste.  
   - Exibe métricas como acurácia, relatório de classificação e matriz de confusão.  
   - **Onde colocar a imagem da Matriz de Confusão**:  
     - No **Notion**, você pode inserir a imagem logo após a explicação sobre a avaliação do modelo.  
     - Exemplo:

       **Matriz de Confusão - Regressão Logística**  
       ![Captura de tela 2025-01-25 174014](https://github.com/user-attachments/assets/8332c228-4de9-438c-bf73-5be720ba6fdf)

4. **Visualiza os Dados em 3D (PCA)**  
   - Reduz os dados para **3 componentes principais** com o PCA.  
   - Gera **gráficos 3D** para comparar classes reais e preditas.  
   - **Onde colocar a imagem da Visualização 3D**:  
     - No **Notion**, após a explicação do PCA, insira a figura que ilustra as classes preditas.  
     - Exemplo:

       **Visualização 3D - Classes Preditas**  
       ![Captura de tela 2025-01-25 174238](https://github.com/user-attachments/assets/a0178f90-0a35-4a03-994e-735bf93f9b22)
---

## 📦 **Dependências e Instalação**
- **Python 3.7+**  
- [**pandas**](https://pandas.pydata.org/)  
- [**seaborn**](https://seaborn.pydata.org/)  
- [**matplotlib**](https://matplotlib.org/)  
- [**scikit-learn**](https://scikit-learn.org/stable/)  

Instale tudo de uma só vez:
```bash
pip install pandas seaborn matplotlib scikit-learn
```
> **Observação**: `mpl_toolkits` (usado no 3D) faz parte do `matplotlib`.

---

## 🏃 **Como Executar o Projeto**
1. **Clone o repositório** ou baixe-o em formato ZIP:
   ```bash
   git clone https://github.com/seu-usuario/nome-do-repositorio.git
   ```
2. **Entre na pasta** do projeto:
   ```bash
   cd nome-do-repositorio
   ```
3. **Instale as dependências**:
   ```bash
   pip install pandas seaborn matplotlib scikit-learn
   ```
4. **Execute o arquivo principal**:
   ```bash
   python main.py
   ```
   > Substitua `main.py` pelo nome do seu arquivo (ex.: `iris_logistic.py`).

5. **Confira o console**:  
   - Você verá logs sobre as etapas: carregamento de dados, escalonamento, `GridSearchCV`, métricas de avaliação etc.

6. **Visualize os Gráficos**:  
   - Uma matriz de confusão em formato de *heatmap*.  
   - Dois gráficos 3D (classes reais e classes preditas).

---
## 🛠️ **Estrutura do Código**
- **`load_and_preprocess_data()`**  
  - Carrega o conjunto Iris, escalona as variáveis e retorna `X_scaled`, `y` e `iris`.  

- **`train_model(X_train, y_train)`**  
  - Define um `Pipeline` e realiza ajuste de hiperparâmetros com `GridSearchCV`.  

- **`evaluate_model(model, X_test, y_test, iris)`**  
  - Calcula as métricas de classificação, exibe a matriz de confusão e imprime relatórios no console.  

- **`visualize_pca(X_test, y_test, y_pred, iris)`**  
  - Reduz os dados a 3 componentes principais e plota gráficos 3D para as classes reais e preditas.  

- **`main()`**  
  - Orquestra a execução: carrega dados, treina, avalia e visualiza.

---

## 🔍 **Possíveis Customizações**
- **Mudar Hiperparâmetros**: Ajuste `C` ou `solver` em `GridSearchCV` para testar novas configurações.  
- **Trocar o Classificador**: Tente `SVC`, `RandomForestClassifier` ou outros algoritmos.  
- **Alterar Componentes PCA**: Defina `n_components=2` para visualizações bidimensionais.
