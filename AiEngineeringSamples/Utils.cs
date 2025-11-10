using System.Text.RegularExpressions;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace AiEngineeringSamples;

/// <summary>
/// Represents a document with text data.
/// </summary>
/// <param name="Text">The text content of the document.</param>
internal record DocumentData(string Text);

/// <summary>
/// Represents a document with vectorized features.
/// </summary>
internal record DocumentVectors 
{ 
    /// <summary>
    /// The vectorized features of the document.
    /// </summary>
    [VectorType] 
    public float[] Features { get; init; } = [0.0f];  
}

internal static class Utils
{
    /// <summary>
    /// Calculates the cosine similarity between two vectors.
    /// </summary>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>The cosine similarity as a float value.</returns>
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

    /// <summary>
    /// Tokenizes a given text into words and punctuation marks.
    /// </summary>
    /// <param name="text">The input text to tokenize.</param>
    /// <returns>A list of tokens (words and punctuation).</returns>
    public static List<string> TokenizeWords(string text)
    {
        // regex: captures words (letters/numbers) and individual punctuation
        return Regex.Matches(text, @"\w+|[^\w\s]")
            .Cast<Match>()
            .Select(m => m.Value)
            .ToList();
    }

    /// <summary>
    /// Splits a given text into sentences.
    /// </summary>
    /// <param name="text">The input text to split.</param>
    /// <returns>A list of sentences.</returns>
    public static List<string> TokenizeSentences(string text)
    {
        // split by sentence-ending punctuation followed by a space
        return Regex.Split(text, @"(?<=[.!?])\s+")
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .Select(s => s.Trim())
            .ToList();
    }

    /// <summary>
    /// Preprocesses a given text by tokenizing and filtering out non-alphanumeric tokens.
    /// </summary>
    /// <param name="text">The input text to preprocess.</param>
    /// <returns>A list of preprocessed tokens.</returns>
    public static List<string> PreprocessText(string text)
    {
        var tokens = TokenizeWords(text.ToLowerInvariant());
        return tokens.Where(word => word.All(char.IsLetterOrDigit)).ToList();
    }

    /// <summary>
    /// Retrieves stop words remover options for Portuguese (Brazil).
    /// </summary>
    /// <returns>An instance of <see cref="CustomStopWordsRemovingEstimator.Options"/> configured with Portuguese stop words.</returns>
    public static IStopWordsRemoverOptions GetStopWordsRemoverOptionsPtBr() =>
        new CustomStopWordsRemovingEstimator.Options
        {
            StopWords = File.ReadAllLines(Path.Combine(Environment.CurrentDirectory, "sw-ptbr.txt")),
        };
}
