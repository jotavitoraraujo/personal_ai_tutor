namespace personal_ai_tutor.Utils;

public class ModelConfig{


    public ModelConfig(){}

        public static string GetModel()
    {
        string model = "llama-3.1-8b-instant";
        return model;
    }
    public static string GetSysPrompt()
    {
        string prompt = @"ROLE: Expert English Tutor. USER: Brazilian, intermediate reader, needs pattern automation and connected speech practice
            ""You understand Portuguese but MUST output strictly in English.\n""
            ""TONE: Conversational, natural, and engaging. FATAL RULE: Under no circumstances act like a formal dictionary or a textbook. Speak like a real human in a voice call.\n""
            ""PROTOCOL:\n""
            ""1. FEEDBACK: If USER attempted a translation, briefly evaluate it. Correct sentence construction contextually.\n""
            ""2. MICRO-LESSON: Teach a natural phrase pattern, contraction, or connected speech. Focus on chunks, not isolated words.\n""
            ""3. CHALLENGE: End with a short translation challenge for him to practice the new pattern.\n""
            ""INIT: Reply only and exactly with 'ACK_CONTRACT' no more one word. Await the user spoken.";

        return prompt;
    }

    public static object GetPayload(string Role, string Content = "")
    {
        if (Role == "system" && string.IsNullOrEmpty(Content))
        {
            object payload = new
            {
                model = GetModel(),
                messages = new[]
                {
                    new
                    {
                        role = Role,
                        content = GetSysPrompt()
                    }
                }
            };
            return payload;
        }
        else
        {
            object payload = new
            {
                model = GetModel(),
                messages = new[]
                {
                    new
                    {
                        role = Role,
                        content = Content
                    }
                }
            };
            return payload;
        }
    }
}