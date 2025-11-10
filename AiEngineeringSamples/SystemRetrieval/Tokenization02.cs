using Microsoft.ML;
using Microsoft.ML.Transforms.Text;

namespace AiEngineeringSamples.SystemRetrieval;

internal static class Tokenization02
{
    private static readonly DocumentData[] Docs =
    [
        new ("Machine learning é um campo da inteligência artificial que permite que computadores aprendam padrões a partir de dados."),
        new ("O aprendizado de máquina dá aos sistemas a capacidade de melhorar seu desempenho sem serem explicitamente programados."),
        new ("Em vez de seguir apenas regras fixas, o machine learning descobre relações escondidas nos dados."),
        new ("Esse campo combina estatística, algoritmos e poder computacional para extrair conhecimento."),
        new ("O objetivo é criar modelos capazes de generalizar além dos exemplos vistos no treinamento."),
        new ("Aplicações de machine learning vão desde recomendações de filmes até diagnósticos médicos."),
        new ("Os algoritmos de aprendizado de máquina transformam dados brutos em previsões úteis."),
        new ("Diferente de um software tradicional, o ML adapta-se conforme novos dados chegam."),
        new ("O aprendizado pode ser supervisionado, não supervisionado ou por reforço, dependendo do tipo de problema."),
        new ("Na prática, machine learning é o motor que impulsiona muitos avanços em visão computacional e processamento de linguagem natural."),
        new ("Mais do que encontrar padrões, o machine learning ajuda a tomar decisões baseadas em evidências.")
    ];    
    
    /// <summary>
    /// Método principal demonstrando pré-processamento de texto, vetorização TF-IDF e busca por similaridade.
    /// Mimica funcionalidades similares ao uso de TF-IDF e busca por similaridade em Python
    /// </summary>
    public static void Main()
    {
        var context = new MLContext();
        var (vectorizer, processedDocs) = context.PreProcess(Docs);
        var vectors = context.GetVectors(processedDocs);
        var sims = context.SearchTfidf([new DocumentData("machine learning")], vectorizer, vectors).ToArray();

        Console.WriteLine($"Top for query: \"machine learning\"");
        foreach (var (idx, sim) in sims.Take(10))
            Console.WriteLine($"Doc {idx:00} -> {sim:F4}: {Docs[idx]}");
    }

    /// <summary>
    /// Pré-processa os documentos, aplicando tokenização, remoção de stopwords e vetorização TF-IDF.
    /// </summary>
    /// <param name="context"></param>
    /// <param name="docs"></param>
    /// <returns></returns>
    private static (ITransformer Vectorizer, IDataView PreProcessedDocs) PreProcess(this MLContext context, DocumentData[] docs)
    {
        var estimator = GetTextFeaturizingEstimator(context);
        var data = context.Data.LoadFromEnumerable(docs);
        var transformer = estimator.Fit(data);
        return (transformer, transformer.Transform(data));
    }

    /// <summary>
    /// Retorna um estimador de featurização de texto com opções específicas para pré-processamento.
    /// </summary>
    /// <param name="context"></param>
    /// <returns></returns>
    private static TextFeaturizingEstimator GetTextFeaturizingEstimator(MLContext context)
    {
        var options = new TextFeaturizingEstimator.Options
        {
            CaseMode = TextNormalizingEstimator.CaseMode.Lower,
            StopWordsRemoverOptions = Utils.GetStopWordsRemoverOptionsPtBr(),
            WordFeatureExtractor = new WordBagEstimator.Options { NgramLength = 1, UseAllLengths = false },
            // activate TF‑IDF internally
            Norm = TextFeaturizingEstimator.NormFunction.L2
        };

        return context.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnNames: nameof(DocumentData.Text), options: options);
    }

    /// <summary>
    /// Retorna a matriz de vetores TF-IDF dos documentos pré-processados.
    /// </summary>
    /// <param name="context"></param>
    /// <param name="preProcessedDocs"></param>
    /// <returns></returns>
    private static float[][] GetVectors(this MLContext context, IDataView preProcessedDocs) =>
        context.Data.CreateEnumerable<DocumentVectors>(preProcessedDocs, reuseRowObject: false)
            .Select(f => f.Features)
            .ToArray();

    /// <summary>
    /// Retorna os documentos mais similares ao query usando vetores TF-IDF e similaridade cosseno.
    /// </summary>
    /// <param name="context"></param>
    /// <param name="query"></param>
    /// <param name="vectorizer"></param>
    /// <param name="source"></param>
    /// <returns></returns>
    private static IEnumerable<(int idx, float sim)> SearchTfidf(this MLContext context, DocumentData[] query, ITransformer vectorizer, float[][] source)
    {
        var data = context.Data.LoadFromEnumerable(query);
        var transform = vectorizer.Transform(data);
        var vectors = context.Data.CreateEnumerable<DocumentVectors>(transform, reuseRowObject: false).First().Features;

        return source.Select((v, idx) => (idx, sim: Utils.CosineSimilarity(v, vectors)))
            .OrderByDescending(x => x.sim)
            .ToArray();
    }
}