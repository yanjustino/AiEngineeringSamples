using AiEngineeringSamples;
using AiEngineeringSamples.SystemRetrieval;

var docs = new[]
{
    new DocumentData("Machine learning é um campo da inteligência artificial que permite que computadores aprendam padrões a partir de dados."),
    new DocumentData("O aprendizado de máquina dá aos sistemas a capacidade de melhorar seu desempenho sem serem explicitamente programados."),
    new DocumentData("Em vez de seguir apenas regras fixas, o machine learning descobre relações escondidas nos dados."),
    new DocumentData("Esse campo combina estatística, algoritmos e poder computacional para extrair conhecimento."),
    new DocumentData("O objetivo é criar modelos capazes de generalizar além dos exemplos vistos no treinamento."),
    new DocumentData("Aplicações de machine learning vão desde recomendações de filmes até diagnósticos médicos."),
    new DocumentData("Os algoritmos de aprendizado de máquina transformam dados brutos em previsões úteis."),
    new DocumentData("Diferente de um software tradicional, o ML adapta-se conforme novos dados chegam."),
    new DocumentData("O aprendizado pode ser supervisionado, não supervisionado ou por reforço, dependendo do tipo de problema."),
    new DocumentData("Na prática, machine learning é o motor que impulsiona muitos avanços em visão computacional e processamento de linguagem natural."),
    new DocumentData("Mais do que encontrar padrões, o machine learning ajuda a tomar decisões baseadas em evidências."),
};


Tokenization02.Main(docs);

