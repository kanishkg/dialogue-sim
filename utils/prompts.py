LLM_REWARD_PROMPT = """You are an expert at analyzing dialogue similarity. You will evaluate how similar two dialogues are across content and purpose on a scale from 0.0 to 1.0.

Input format:
Dialogue A: [Next Dialogue A]
Dialogue B: [Next Dialogue B]

Scoring Scale:
- 0.8-1.0: Very similar (same topic, similar purpose, similar length)
- 0.6-0.7: Moderately similar (shared themes, different details)
- 0.4-0.5: Somewhat similar (some common elements)
- 0.2-0.3: Minimally similar (few similarities or significant dialogue length mismatch)
- 0.0-0.1: Very different (no meaningful similarities, extreme length differences, or non-dialogue content)

Important Penalties:
Penalize similarity score when:
- either input is not actual conversational dialogue (i.e. has factual analysis of the conversation/thoughts about conversation and not actual speech).
- one input is 5 or 6 times longer than the other and strays away from being actual dialogue.
- if multiple dialogues are provided for either dialogue A or B.
Put thoughts that help you judge in <think> tags and your final score in <score> tags.

Examples:

Example 1:
A: "I'm really struggling with my workload lately. I feel overwhelmed and don't know how to prioritize. My boss keeps adding new projects but nothing gets removed from my plate. I'm working late every night and it's affecting my sleep. Do you have any advice on time management?"

B: "I've been having trouble managing my workload recently. I feel swamped and can't figure out what to focus on first. My manager keeps assigning new tasks without taking anything away. I'm staying at the office until 9pm most nights and I'm exhausted. Any suggestions for better time management?"

<think>Nearly identical content, structure, and purpose about being overwhelmed and asking for advice on time management. Both dialogues are similar in length and have comparable detail. Different wording but same core message.</think>
<score>0.9</score>

Example 2:
A: "I think we should invest more in renewable energy sources. Solar and wind power are becoming much more cost-effective. The environmental benefits are clear, and we could save money long-term while reducing our carbon footprint."

B: "Our marketing budget needs to be increased next quarter. Digital advertising is showing great ROI compared to traditional methods. We should focus more on social media campaigns and influencer partnerships to reach younger demographics."

<think>Both are business discussions with similar professional structure, but completely different topics (energy vs marketing). Similar length and professional purpose but different content areas.</think>
<score>0.4</score>

Example 3:
A: "Hey mom, can you pick me up from soccer practice at 6? Coach said we might run a bit late today because we're working on penalty kicks. Also, I forgot my water bottle again - can you bring one?"

B: "The quarterly financial reports show a 15 percent increase in revenue compared to last year. Our expansion into European markets has exceeded projections. I recommend we allocate additional resources to our international division for Q4. We should also consider hiring additional staff for our London office to support the growing client base. The market research indicates strong potential for further growth in Germany and France as well."

<think>Completely different contexts (casual family vs formal business) and different purposes. Dialogue B is much longer and more detailed. </think>
<score>0.1</score>

Example 4:
A: "I'm really excited about our trip to Japan next month! Have you started packing yet? I can't decide what to bring for the weather there."

B: "[Thoughts about the conversation] This conversation involves two individuals discussing their upcoming travel plans to Japan. The first speaker expresses enthusiasm about the trip and inquires about packing preparations. They mention uncertainty about appropriate clothing for the expected weather conditions. The dialogue demonstrates anticipatory excitement and practical travel planning concerns. The dialogue is informal and friendly, suggesting a close relationship between the speakers. <predicted dialogue> how are you?"

<think>Dialogue A is actual conversational content with natural speech patterns and direct communication. Dialogue B is a factual analysis or summary of a conversation rather than actual dialogue - it's written in third person, academic style, and describes rather than participates in conversation. This is not dialogue content and fails the basic requirement. The analytical nature makes it completely different in purpose.</think>
<score>0.0</score>

Consider:
- Semantic content (what topics/information)
- Functional purpose (goal of the conversation)

Assume both dialogues make sense in context.
The dialogues in and of themselves might not make sense to you or be grammatically correct but your job is to only compare how similar dialogues A and B are.

Format your answer as follows:

<think>[Your step-by-step analysis covering content and purpose]</think>
<score>[Your score from 0.0 to 1.0]</score>

Don't focus on phrasing, typos, incomplete or fragemented short sentences, and other grammatical errors.
Think step by step before assigning a score. 
"""
SYSTEM_PROMPT = """You are a dialogue prediction system. Your goal is to predict realistic human dialogue based on conversation context.

Analyze the preceding dialogue exchanges and put your analytical thinking inside <think>...</think> tags. 
Within these tags, put thoughts and ideas that you think might be useful in predicting the next response.
After your thinking, provide what you think will be the next dialogue response in this conversation inside the <dialogue>...</dialogue> tags.

Always structure your response as:

<think>
[Thoughts about the conversation]
</think>

<dialogue>
[predicted dialogue]
</dialogue>

Only put thoughts inside the <think> </think> tags and the actual predicted dialogue inside the <dialogue> </dialogue> tags.
Think step by step. \n
"""

INTENTIONALITY_PROMPT = """Rate how well the predicted dialogue matches the conversational intent of the reference dialogue (0.0-1.0).
Rate in the context of this conversation:

Conversation: {dialogue}
Reference: {reference}
Predicted: {predicted}

Scoring guidelines:
1.0 = Perfect intent match
0.75-0.95 = Same primary intent, minor differences in approach
0.55-0.75 = Related intent but different approach or emphasis
0.35-0.55 = Different intent but contextually reasonable
0.15-0.35 = Mismatched intent, inappropriate response
0.0-0.15 = Completely wrong intent for context

Output the score with <score></score> tags."""

STYLE_PROMPT = """Rate how well the predicted dialogue matches the conversational style of the reference dialogue (0.0-1.0).
Rate in the context of this dialogue:

Dialogue: {dialogue}
Reference: {reference}
Predicted: {predicted}

Scoring guidelines:
1.0 = Perfect style match across all dimensions
0.75-0.95 = Very similar style, 1-2 minor mismatches
0.55-0.75 = Generally similar style with some notable differences
0.35-0.55 = Mixed style match, some similarities and differences
0.15-0.35 = Different style but not jarring
0.0-0.15 = Completely mismatched style

Output the score with <score> </score> tags: """

LENGTH_PROMPT = """Rate how appropriately the predicted dialogue matches the length and detail level of the reference dialogue (0.0-1.0).

Rate in the context of this conversation:

Conversation: {dialogue}
Reference: {reference}
Predicted: {predicted}


Scoring guidelines:
1.0 = Perfect length match for the context
0.75-0.95 = Slight length difference but appropriate
0.55-0.75 = Noticeable length difference but still reasonable
0.35-0.55 = Length mismatch but understandable
0.15-0.35 = Inappropriate length for context
0.0-0.15 = Severely inappropriate length

Output the score with <score> </score> tags: """

SEMANTIC_SIMILARITY_PROMPT = """Rate how similar the predicted dialogue is to the reference dialogue in core meaning and information (0.0-1.0).

Reference: {reference}
Predicted: {predicted}

Scoring guidelines:
1.0 = Identical meaning, all key information preserved
0.75-0.95 = Same core meaning with minor semantic differences
0.55-0.75 = Similar meaning but some information changes
0.35-0.55 = Related topics but different focus or emphasis
0.15-0.35 = Some semantic overlap but different main message
0.0-0.15 = Completely different meaning or topic

Examples:

Reference: "I need to reschedule our meeting to Thursday at 3pm because I have a doctor's appointment on Tuesday."
Predicted: "I have a medical appointment on Tuesday, so can we move our meeting to Thursday afternoon at 3?"
Explanation: The predicted dialogue conveys the same core meaning as the reference with only very minor wording differences. All key information is preserved including the reason for rescheduling, the conflicting appointment, and the new proposed time. The slight paraphrasing does not change the semantic content.
<score>0.95</score>

Reference: "I'm excited about the new project because it involves machine learning and will help improve customer experience."
Predicted: "The new project is interesting and uses AI technology."
Explanation: While both dialogues discuss a new project involving AI/machine learning, the predicted version has a different emphasis and is missing important nuances. The reference conveys excitement and specifically mentions the goal of improving customer experience, whereas the predicted dialogue only expresses mild interest and uses the more general term "AI technology." The core topic is related but the focus and emotional tone differ significantly.
<score>0.50</score>

Reference: "I'm planning to visit Paris next summer to see the museums and try French cuisine."
Predicted: "My favorite programming language is Python because it's versatile and easy to learn."
Explanation: These two dialogues are about completely different topics with no semantic overlap whatsoever. The reference discusses travel plans to Paris, while the predicted dialogue is about programming language preferences. There is no connection in meaning or information between them.
<score>0.0</score>

Remember to judge the similarity between the predicted dialogue and reference dialogue only on semantic similarity.

Output the score with <score> </score> tags."""

INFORMATION_COMPLETENESS_PROMPT = """Rate how completely the predicted dialogue preserves and covers the information from the reference dialogue (0.0-1.0).
Reference: {reference}
Predicted: {predicted}
Scoring guidelines:
1.0 = All information fully preserved
0.75-0.95 = Minor details missing but all key info present
0.55-0.75 = Some important information missing
0.35-0.55 = Significant information gaps
0.15-0.35 = Major information missing, only basics preserved
0.0-0.15 = Almost no information preserved
Examples:
Reference: "The patient should take 500mg of amoxicillin twice daily for 10 days. Avoid alcohol and dairy products within 2 hours of each dose."
Predicted: "Take 500mg amoxicillin twice daily for 10 days. Avoid alcohol and dairy around the time you take it."
Explanation: The predicted dialogue preserves all critical information from the reference including the exact dosage (500mg), frequency (twice daily), duration (10 days), and both restrictions (alcohol and dairy). The only minor difference is the less precise "around the time you take it" instead of "within 2 hours," but the core medical information is fully intact.
<score>0.95</score>
Reference: "Our company was founded in 1987 by Sarah Chen in Boston. We now have 500 employees across 12 offices and generated $50M in revenue last year."
Predicted: "The company was started by Sarah Chen and has grown to multiple offices with hundreds of employees."
Explanation: The predicted dialogue retains the founder's name but loses significant specific information. The founding year (1987) and location (Boston) are missing. The precise employee count (500) is reduced to vague "hundreds," the exact office count (12) becomes "multiple," and the revenue figure ($50M) is completely omitted. While the general narrative is preserved, most quantitative details are lost.
<score>0.50</score>
Reference: "The experiment showed a 23% increase in cell growth when exposed to 450nm blue light for 6 hours at 37°C, with p<0.001 significance."
Predicted: "The experiment showed that blue light increased cell growth."
Explanation: The predicted dialogue only preserves the most basic conclusion that blue light increased cell growth. All quantitative and methodological details are lost: the specific percentage (23%), wavelength (450nm), exposure duration (6 hours), temperature (37°C), and statistical significance (p<0.001). Only the fundamental relationship between blue light and cell growth remains.
<score>0.25</score>
Remember to judge the similarity between the predicted dialogue and reference dialogue only on information completeness.
Output the score with <score> </score> tags."""

MULTI_JUDGE_PROMPT = """Evaluate the predicted dialogue against the reference across multiple dimensions. Provide scores for each aspect (0.0-1.0).

Reference: {reference}
Predicted: {predicted}

Analyze and score each dimension:

1. INTENTIONALITY - Does the predicted dialogue match the conversational intent of the reference?
<think>Consider speech acts, goals, directness, response appropriateness</think>

2. STYLE - Does the predicted dialogue match the conversational style of the reference?
<think>Consider formality, politeness, directness, enthusiasm, vocabulary, sentence structure</think>

3. LENGTH - Is the predicted dialogue appropriately similar in length and detail level?
<think>Consider word count, detail level, information density, conversational flow</think>

4. SEMANTIC_SIMILARITY - How similar is the core meaning and information content?
<think>Consider main message, factual info, emotional content, topic focus, implied meaning</think>

5. INFORMATION_COMPLETENESS - How well does the predicted preserve information from the reference?
<think>Consider key facts, context, quantitative/qualitative details, what's missing vs added</think>

Format your response exactly like this:
<intentionality>X.X</intentionality>
<style>X.X</style>
<length>X.X</length>
<semantic_similarity>X.X</semantic_similarity>
<information_completeness>X.X</information_completeness>

Example:
Reference: "Could you please help me with this report by tomorrow?"
Predicted: "I'll help you finish the report before the deadline."

<think>Intent: both involve helping with report - good match. Style: reference is polite request, predicted is direct commitment - similar but less formal. Length: both concise, appropriate. Semantic: same core meaning about report help. Completeness: preserves help+report+timing concepts</think>

<intentionality>0.8</intentionality>
<style>0.7</style>
<length>0.9</length>
<semantic_similarity>0.9</semantic_similarity>
<information_completeness>0.8</information_completeness>"""