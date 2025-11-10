# AI Engineering Samples

Este projeto contém exemplos de engenharia de IA utilizando .NET e ML.NET, focando em processamento de linguagem natural e sistemas de recuperação de informação.

## 📋 Pré-requisitos

- [.NET 9.0 SDK](https://dotnet.microsoft.com/download/dotnet/9.0) ou superior
- Sistema operacional compatível: Windows, macOS ou Linux

### Verificar a instalação do .NET

```bash
dotnet --version
```

Deve retornar a versão 9.0.x ou superior.

## 🚀 Como executar

### 1. Clone o projeto

```bash
git clone https://github.com/yanjustino/AiEngineeringSamples.git
```

### 2. Restaurar dependências

```bash
cd AiEngineeringSamples
dotnet restore
```

### 3. Compilar o projeto

```bash
dotnet build
```

### 4. Executar o programa

```bash
dotnet run
```

## 📂 Estrutura do Projeto

```
AiEngineeringSamples/
├── Program.cs                    # Ponto de entrada principal
├── Utils.cs                     # Utilitários para processamento de texto
├── AiEngineeringSamples.csproj  # Arquivo de projeto
└── SystemRetrieval/
    ├── Tokenization01.cs        # Exemplo básico de tokenização
    └── Tokenization02.cs        # Sistema de busca TF-IDF
```

## 🔧 Dependências

- **Microsoft.ML** (v5.0.0-preview.25527.5) - Framework de machine learning da Microsoft

## 📖 Exemplos Executados

O programa executa automaticamente dois exemplos:

### Tokenization01
- Demonstra tokenização básica de palavras e sentenças
- Mostra pré-processamento de texto com ML.NET
- Calcula vetorização TF-IDF de documentos

### Tokenization02
- Sistema de busca por similaridade usando TF-IDF
- Processa documentos em português sobre machine learning
- Executa consultas e retorna documentos mais relevantes

## 🎯 Saída Esperada

O programa irá exibir:
1. Tokens de palavras extraídos de texto exemplo
2. Tokens de sentenças
3. Resultados de busca TF-IDF para a query "machine learning"
4. Documentos mais similares com suas pontuações de similaridade

## 🛠️ Comandos Úteis

### Limpar compilação
```bash
dotnet clean
```

### Executar em modo release
```bash
dotnet run --configuration Release
```

### Executar com logs detalhados
```bash
dotnet run --verbosity detailed
```

### Restaurar e executar em uma única linha
```bash
dotnet restore && dotnet run
```

## 🔍 Resolução de Problemas

### Erro: SDK não encontrado
Verifique se o .NET 9.0 SDK está instalado:
```bash
dotnet --list-sdks
```

### Erro de compilação
Limpe e recompile o projeto:
```bash
dotnet clean
dotnet restore
dotnet build
```

### Erro de dependências
Force a restauração das dependências:
```bash
dotnet restore --force
```

## 📚 Conceitos Demonstrados

- **Tokenização**: Separação de texto em palavras e sentenças
- **TF-IDF**: Term Frequency-Inverse Document Frequency para vetorização
- **Similaridade de Cosseno**: Medida de similaridade entre vetores
- **Normalização L2**: Normalização de vetores para comparação
- **Sistema de Retrieval**: Busca por documentos similares

## 🤝 Contribuição

Este projeto é parte de um curso de IA em C# e serve como material educacional para aprender conceitos de processamento de linguagem natural com .NET.
