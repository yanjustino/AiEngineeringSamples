using Microsoft.ML;
using Microsoft.ML.Transforms.Text;

namespace AiEngineeringSamples.SystemRetrieval;

internal static class Tokenization02
{
    public static void Main()
    {
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
        
        var context = new MLContext();
        var (vectorizer, processedDocs) = context.PreProcess(docs);
        var vectors = context.GetVectors(processedDocs);
        var query = new[] { new DocumentData("machine learning") };
        var sims = context.SearchTfidf(query, vectorizer, vectors).ToArray();

        Console.WriteLine($"Top for query: \"machine learning\"");
        foreach (var (idx, sim) in sims.Take(10))
            Console.WriteLine($"Doc {idx} -> {sim:F4}: {docs[idx]}");        
    }
    
    private static (ITransformer Vectorizer, IDataView PreProcessedDocs) PreProcess(this MLContext context, DocumentData[] docs)
    {
        var estimator = GetTextFeaturizingEstimator(context);
        var data = context.Data.LoadFromEnumerable(docs);
        var transformer = estimator.Fit(data);
        return (transformer, transformer.Transform(data));
    }

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

    private static float[][] GetVectors(this MLContext context, IDataView preProcessedDocs)
    {
        var vectors = context.Data.CreateEnumerable<DocumentVectors>(preProcessedDocs, reuseRowObject: false)
            .Select(f => f.Features)
            .ToArray();

        return vectors;
    }

    private static IEnumerable<(int idx, float sim)> SearchTfidf(this MLContext context, DocumentData[] query, ITransformer vectorizer, float[][] source)
    {
        var data = context.Data.LoadFromEnumerable(query);
        var transform = vectorizer.Transform(data);
        var vectors = context.Data.CreateEnumerable<DocumentVectors>(transform, reuseRowObject:false).First().Features;

        return source.Select((v, idx) => (idx, sim: Utils.CosineSimilarity(v, vectors)))
            .OrderByDescending(x => x.sim)
            .ToArray();
    }
}