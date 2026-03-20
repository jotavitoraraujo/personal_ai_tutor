using personal_ai_tutor.Engines;

LlmEngine LlmEngine = new LlmEngine();
string responseToContract = await LlmEngine.SendPersonaAsync();
Console.WriteLine(responseToContract);
if (responseToContract == "ACK_CONTRACT")
{
    string pathToRecord = @"C:\Users\Lider CPD\Documents\Sound Recordings\Recording.wav";
    string transcription = await LlmEngine.SendTranscriptionAsync(pathToRecord);
    string finalResponse = await LlmEngine.SendPromptAsync(transcription);
    Console.WriteLine("\n");
    Console.WriteLine($"[TRANSCRIPTION] ::: {transcription}\n");
    Console.WriteLine($"[RESPONSE] ::: {finalResponse}\n");
    Console.WriteLine("\n");
}
else
{
    throw new Exception();
}



