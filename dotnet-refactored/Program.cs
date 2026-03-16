using personal_ai_tutor.Engines;

var LlmEngine = new LlmEngine();
var jsonResponse = await LlmEngine.SendPromptAsync("Reply just with 'TEST'");
Console.WriteLine(jsonResponse);