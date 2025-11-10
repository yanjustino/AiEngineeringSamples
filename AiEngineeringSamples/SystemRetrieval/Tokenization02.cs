using Microsoft.ML;
using Microsoft.ML.Transforms.Text;

namespace AiEngineeringSamples.SystemRetrieval;

internal static class Tokenization01
{
    public static void Main(DocumentData[] docs)
    {
        var ml = new MLContext();
     
    }    
}

internal static class Tokenization02
{
    public static void Main(DocumentData[] docs)
    {
        var ml = new MLContext();
        var pipeline = ml.PreProcess(docs);
        var features = ml.GetMatrix(pipeline.PreProcessedDocs);
        var sims = ml.SearchTfidf(pipeline.Vectorizer, features).ToArray();

        Console.WriteLine($"Top for query: \"machine learning\"");
        foreach (var (idx, sim) in sims.Take(10))
            Console.WriteLine($"Doc {idx} -> {sim:F4}: {docs[idx]}");        
    }
    
    private static (ITransformer Vectorizer, IDataView PreProcessedDocs) PreProcess(this MLContext context, DocumentData[] docs)
    {
        var options = new TextFeaturizingEstimator.Options
        {
            CaseMode = TextNormalizingEstimator.CaseMode.Lower,
            StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options(),
            WordFeatureExtractor = new WordBagEstimator.Options { NgramLength = 1, UseAllLengths = false },
            // activate TF‑IDF internally
            Norm = TextFeaturizingEstimator.NormFunction.L2
        };

        var estimator = context.Transforms.Text.FeaturizeText(outputColumnName: "Features",
            inputColumnNames: nameof(DocumentData.Text), options: options);
        
        var data = context.Data.LoadFromEnumerable(docs);
        var transformer = estimator.Fit(data);
        return (transformer, transformer.Transform(data));
    }

    private static float[][] GetMatrix(this MLContext context, IDataView preProcessedDocs)
    {
        var vectors = context.Data.CreateEnumerable<DocumentVectors>(preProcessedDocs, reuseRowObject: false)
            .Select(f => f.Features)
            .ToArray();

        return vectors;
    }

    private static IEnumerable<(int idx, float sim)> SearchTfidf(this MLContext context, ITransformer vectorizer, float[][] vectors)
    {
        var query = new[] { new DocumentData("machine learning") };
        var data = context.Data.LoadFromEnumerable(query);
        var transform = vectorizer.Transform(data);
        var qFeat = context.Data.CreateEnumerable<DocumentVectors>(transform, reuseRowObject:false).First().Features;

        return vectors.Select((v, idx) => (idx, sim: Utils.CosineSimilarity(v, qFeat)))
            .OrderByDescending(x => x.sim)
            .ToArray();
    }
}