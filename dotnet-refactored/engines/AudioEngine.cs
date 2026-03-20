namespace personal_ai_tutor.Engines;

using System.IO;

public sealed class AudioEngine
{
    public AudioEngine()
    {
        
    }

    public async Task<float[]> ReadAndCastWavAsync(string path)
    {
        Memory<byte> rawBytes = await File.ReadAllBytesAsync(path);
        Memory<byte> bytesWithoutHeaders = rawBytes.Slice(44);
        float[] floatArr = ConvertPcmToFloat(bytesWithoutHeaders.Span);
        return floatArr;
    }

    public float[] ConvertPcmToFloat(ReadOnlySpan<byte> windowArr)
    {
        int lengthToArr = windowArr.Length / 4;
        float[] floatArr = new float[lengthToArr];
        
        for (int i = 0; i < lengthToArr; i++)
        {
            int byteIndex = i * 4;
            short sample = BitConverter.ToInt16(windowArr.Slice(byteIndex, 2));
            float sampleConvToFloat = sample / 32768.0f;
            floatArr[i] = sampleConvToFloat;
        }

        return floatArr;
    }
}   