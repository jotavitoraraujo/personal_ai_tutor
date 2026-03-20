namespace personal_ai_tutor.Engines;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using personal_ai_tutor.Utils;

public class LlmEngine
{
    private readonly HttpClient Client;
    public LlmEngine()
    {
        Client = new HttpClient();
        Client.BaseAddress = new Uri("https://api.groq.com/openai/v1/");
        string? apiKey = Environment.GetEnvironmentVariable("GROQ_API_KEY");
        
        if (apiKey is not null)
        {
            Client.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
        }
        else
        {
            throw new Exception("[ERRO] The API Key is empty.");
        };
    }

    private string ParseJsonResponse(string jsonPromptResponse)
    {
        using JsonDocument JsonDocumentParsed = JsonDocument.Parse(jsonPromptResponse);
        string finalResponse = JsonDocumentParsed.RootElement.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString() ?? "[WARNING] Response returned as NULL";
        return finalResponse;
    }

    private string ParseTranscriptionResponse(string jsonTranscriptionResponse)
    {
        using JsonDocument JsonDocumentParsed = JsonDocument.Parse(jsonTranscriptionResponse);
        JsonElement finalTranscription = JsonDocumentParsed.RootElement.GetProperty("text");
        return finalTranscription.ToString();
    }

    private async Task<string> SendPayloadToLLMAsync(object obj)
    {  
        string jsonString = JsonSerializer.Serialize(obj);
        StringContent httpContent = new StringContent(jsonString, Encoding.UTF8, "application/json");
        HttpResponseMessage httpResponse = await Client.PostAsync("chat/completions", httpContent);
        string stringResponse = await httpResponse.Content.ReadAsStringAsync();
        string finalResponse = ParseJsonResponse(stringResponse);
        return finalResponse;
    }
            
    private async Task<string> SendTranscriptionToWhisperAsync(string path)
    {        
        byte[] rawBytes = await File.ReadAllBytesAsync(path);
        using var fileContent = new ByteArrayContent(rawBytes);
        using MultipartFormDataContent Form = new MultipartFormDataContent();
        Form.Add(fileContent, "file", "audio.wav");
        Form.Add(new StringContent("whisper-large-v3"), "model");
        HttpResponseMessage httpResponse = await Client.PostAsync("audio/transcriptions", Form);
        string jsonTranscriptionResponse = await httpResponse.Content.ReadAsStringAsync();
        string finalTranscription = ParseTranscriptionResponse(jsonTranscriptionResponse);
        return finalTranscription;
    }

    public async Task<string> SendPersonaAsync()
    {
        object payload = ModelConfig.GetPayload("system");
        string finalResponse = await SendPayloadToLLMAsync(payload);
        return finalResponse;
    }

    public async Task<string> SendPromptAsync(string prompt)
    {
        object payload = ModelConfig.GetPayload("user", prompt);
        string finalResponse = await SendPayloadToLLMAsync(payload);
        return finalResponse;
    }

    public async Task<string> SendTranscriptionAsync(string path)
    {
        string finalTranscription = await SendTranscriptionToWhisperAsync(path);
        return finalTranscription;
    }
}