using Microsoft.ML;

namespace AiEngineeringSamples.SystemRetrieval;

internal static class Tokenization01
{
    /// <summary>
    /// Método principal demonstrando tokenização de palavras e sentenças, além de pré-processamento de texto.
    /// Mimica funcionalidades similares ao uso de NLTK em Python
    /// </summary>
    public static void Main()
    {
        var text = "Machine learning is fascinating. It enables computers to learn from data. Let's explore it! It's amazing.";

        // tokenização de palavras (equivalente a nltk.word_tokenize)
        var wordTokens = Utils.TokenizeWords(text);
        Console.WriteLine("Word tokens:");
        Console.WriteLine("[" + string.Join(", ", wordTokens.Select(w => $"'{w}'")) + "]");
        Console.WriteLine();

        // tokenização de sentenças (equivalente a nltk.sent_tokenize)
        var sentenceTokens = Utils.TokenizeSentences(text);
        Console.WriteLine("Sentence tokens:");
        Console.WriteLine("[" + string.Join(", ", sentenceTokens.Select(s => $"'{s}'")) + "]");
        Console.WriteLine();

        // documentos
        var documents = new[]
        {
            "Natural Language Processing is a field of AI.",
            "It focuses on the interaction between computers and humans.",
            "Applications include chatbots, translation, and sentiment analysis.",
        };

        // pré-processamento
        var preprocessedDocs = documents
            .Select(doc => string.Join(" ", Utils.PreprocessText(doc)))
            .ToList();

        Console.WriteLine("Preprocessed documents:");
        Console.WriteLine("[" + string.Join(", ", preprocessedDocs.Select(d => $"'{d}'")) + "]");
    } 
}