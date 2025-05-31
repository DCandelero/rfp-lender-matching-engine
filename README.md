# Nome do projeto

## Objetivo do projeto
*\<Descreva o projeto em algumas linhas\>*

## Setup e instalação de dependências
*\<Descrição detalhada dos passos para instalar as dependencias\>*

Após dar o clone inicial no repositório, adicione o path do seu diretório no arquivo _config.py_. 
Com isso, você poderá referenciar as pastas através das variáveis de PATH.
```
# Adicione o caminho nessa lista
PROJ_PATH_LIST = [r'<PROJECT_PATH>']
```

### Ambiente virutal

Em seguida, é recomendado que crie um ambiente virtual nomeado 'env' (Pois já está configurado para não ser versionado no arquivo .gitignore). 

- Passo 1: No diretório do projeto, abra um terminal integrado e inicie o ambiente virtual (Obs: Deve se ter instalado o python em sua máquina local!)
    ```
    $ python -m venv env
    ```

- Passo 2: Após executar o comando, irá ser criado um novo diretório chamado 'env'. Agora, para iniciar o ambiente virtual deve-se utilizar o seguinte comando:
    ```
    # Windows
    $ .\env\Scripts\activate

    # Linux ou Mac
    $ source env/bin/activate
    ```

    No Windows, caso o sistema não permita ativar o ambiente virtual, será necessário permitir a execução via linha de comando. Abra o PowerShell como admnistrador, execute o código abaixo e digite 'A', para sempre permitir a execução de scripts:
    ```
    $ Set-ExecutionPolicy AllSigned
    ```

- Passo 3: Com o ambiente virtual ativado, faça o download das dependências e confira se foram devidamente instaladas
    ```
    # Intalar dependências
    $ pip install -r requirements.txt

    # Conferir dependências instaladas
    $ pip freeze
    ```
    Nesse momento, todas as funções e arquivos podem ser executados sem se preocupar com conflitos entre dependências de outros projetos.

- Passo 4: Para desativar o ambiente virtual, basta abrir novamente o terminal integrado e digitar o seguinte comando
    ```
    $ deactivate
    ```
### Anaconda
Uma outra opção é utilizar o gerenciador de ambientes e pacotes Anaconda.
Para isso é necessário primeiro fazer a instalação da [Anaconda](https://www.anaconda.com/download/).

- Passo 1: No terminal do Anaconda crie um ambiente para o projeto. É indicado já especificar a versão do Python a ser utilizada.

    ```
    $ conda create --name (-n) <ENV_NAME> python=<VERSION>
    ```

- Passo 2: Ative o ambiente do projeto

    ```
    $ conda activate <ENV_NAME>
    ```

- Passo 3: Com o ambiente virtual ativado, faça o download das dependências necessárias.

    ```
    # Intalar dependências
    $ pip install -r requirements.txt

    # Conferir dependências instaladas
    $ pip freeze
    ```

- Passo 4: Para desativar o ambiente virtual, basta utilizar o seguinte comando:

    ```
    $ conda deactivate
    ```


## Fonte dos dados
*\<Cite todas as fontes de dados com links para os dados originais\>*

### Aquisição dos dados
*\<Detalhes sobre como foi realizada a aquisição dos dados\>*

### Pré-processamento dos dados
*\<Detalhes sobre o pré-processamento dos dados\>*

## Estrutura do projeto
*\<Explique a estrutura do projeto, detalhando arquivos importantes\>*

```
📦Project
 ┣ 📂data
 ┃ ┣ 📂01-raw               <- Dados originais
 ┃ ┣ 📂02-interim           <- Dados intermediarios com alguma transformação
 ┃ ┗ 📂03-processed         <- Dados finais que serão utilizados no modelo
 ┃
 ┣ 📂models                 <- Modelos treinados
 ┃
 ┣ 📂notebook               <- Jupyter Notebooks
 ┃
 ┣ 📂reports                
 ┃ ┗ 📂figures              <- Gráficos e figuras gerados
 ┃
 ┣ 📂src
 ┃ ┣ __init__.py
 ┃ ┣ 📂data                 <- Scripts de download de dados
 ┃ ┣ 📂features             <- Scripts que transformam os dados originais em features para o modelo
 ┃ ┣ 📂models               <- Scripts para treinamento de modelos e para fazer predições 
 ┃ ┗ 📂visualization        <- Scripts para criação de visualizações
 ┃
 ┣ README.md
 ┣ requirements.txt
 ┗ config.py
```
 

## Resultados
*\<Resumo dos resultados incluindo metricas relevantes e gráficos\>*
