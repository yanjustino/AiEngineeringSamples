using System.Text.RegularExpressions;
using Microsoft.ML.Data;

namespace AiEngineeringSamples;

internal record DocumentData(string Text);
internal record DocumentVectors { [VectorType] public float[] Features { get; init; } = [0.0f];  }

internal static class Utils
{
    public static float CosineSimilarity(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (var i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return (float)(dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12));
    }    
    
    public static List<string> TokenizeWords(string text)
    {
        // regex: captura palavras (letras/números) e pontuação individual
        return Regex.Matches(text, @"\w+|[^\w\s]")
            .Cast<Match>()
            .Select(m => m.Value)
            .ToList();
    }

    // equivalente a nltk.sent_tokenize
    public static List<string> TokenizeSentences(string text)
    {
        // split por pontos de fim de sentença seguidos de espaço
        return Regex.Split(text, @"(?<=[.!?])\s+")
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .Select(s => s.Trim())
            .ToList();
    }

    // equivalente a preprocess_text
    public static List<string> PreprocessText(string text)
    {
        var tokens = TokenizeWords(text.ToLowerInvariant());
        return tokens.Where(word => word.All(char.IsLetterOrDigit)).ToList();
    }    
}