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
}