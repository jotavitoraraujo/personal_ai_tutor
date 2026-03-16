namespace personal_ai_tutor.Engines;

using System.Net;
using System.Net.Http;
using System.Reflection.Metadata.Ecma335;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.VisualBasic;

public class LlmEngine
{
    private readonly HttpClient Client;
    public LlmEngine()
    {
        Client = new HttpClient();
        Client.BaseAddress = new Uri("https://api.groq.com/openai/v1/");
        Client.DefaultRequestHeaders.Add("Authorization", $"Bearer {Environment.GetEnvironmentVariable("GROQ_API_KEY")}" ?? "[ERROR] API KEY NOT FOUNDED.");
    }

    private string ParseJsonResponse(string jsonResponse)
    {
        using JsonDocument JsonDocumentParsed = JsonDocument.Parse(jsonResponse);
        string finalResponse = JsonDocumentParsed.RootElement.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString() ?? "[WARNING] Response returned as NULL";
        return finalResponse;
    }
    
    public async Task<string> SendPromptAsync(string prompt)
    {
        var payload = new
        {
            model = "llama-3.1-8b-instant",
            messages = new[] {
                new {
                    role = "user",
                    content = prompt
                }
            }
        };

        var jsonString = JsonSerializer.Serialize(payload);
        var httpContent = new StringContent(jsonString, Encoding.UTF8, "application/json");
        var responseHttp = await Client.PostAsync("chat/completions", httpContent);
        var jsonResponse = await responseHttp.Content.ReadAsStringAsync();
        string finalResponse = ParseJsonResponse(jsonResponse);
        return finalResponse;
    }

}