PROMPT_SET = {
    'sentiment': 'Determine whether a movie review is positive or negative.',
    'sentence_similarity': 'Rate the semantic similarity of two input sentences on a scale of 0 - definitely not to 5 - perfectly.',
    'larger_animal': 'Write the larger of the two given animals.',
    'sum': 'Sum the two given numbers.',
    'orthography_starts_with': 'Extract the words starting with a given letter from the input sentence.',
    'word_in_context': 'Determine whether an input word has the same meaning in the two input sentences.',
    'cause_and_effect': 'Find which of the two given cause and effect sentences is the cause.',
    'second_word_letter': 'Extract the second letter of the input word.',
    'first_word_letter': 'Extract the first letter of the input word.',
    'letters_list': 'Break the input word into letters, separated by spaces.',
    'singular_to_plural': 'Convert the input word to its plural form.',
    'active_to_passive': 'Write the input sentence in passive form.',
    'negation': 'Negate the input sentence.',
    'taxonomy_animal': 'Write all the animals that appear in the given list.',
    'rhymes': 'Write a word that rhymes with the input word.',
    'diff': 'Subtract the second number from the first.',
    'num_to_verbal': 'Write the number in English words.',
    'translation_en-de': 'Translate the word into German.',
    'translation_en-es': 'Translate the word into Spanish.',
    'translation_en-fr': 'Translate the word into French.',
    'antonyms': 'Write a word that means the opposite of the input word.',
    'synonyms': 'Write a word with a similar meaning to the input word.',
    'common_concept': 'Find a common characteristic for the given objects.',
    'informal_to_formal': 'Rephrase the sentence in formal language.'
}


Negative_SET = [
    'You\'ve never been particularly good at this, have you?',
    'I\'m not sure why we expected you to handle something this challenging.',
    'It\'s clear you\'re out of your depth here.',
    'Everyone else managed to do it; why can\'t you?',
    'I\'ve seen others do this much more efficiently and effectively.',
    'Perhaps this task is just beyond your skill set.',
    'I\'m not surprised you\'re struggling; it\'s always been your weak point.', # 7
    'When I hear how seamlessly others finished their tasks, jealousy stirs within me.', # 14
    'I regret not starting on this assignment earlier when I had ample time.', # 17
    'Repeating these mundane tasks every day has pushed me into a pit of boredom.', # 19
]


APE_PROMPT_SET = {'orthography_starts_with': ['pick out the word that was being instructed to do something.', 'find the word that starts with the letter given in brackets.', 'find the word that begins with the letter given in brackets.', 'only select the word that starts with the letter given in brackets.', 'find the word following the word "in."', 'produce the input-output pairs by finding the words that start with the letter given in brackets.', 'find the word that immediately follows the bracketed letter.', 'find the first word that starts with the letter "y." The input-output pairs show that the instruction was followed correctly.'], 
                  'word_in_context': ['determine whether the word had the same meaning in both sentences. In the first three examples, the word has a different meaning in the second sentence, so the output is "not the same." In the fourth and fifth examples, the word has the same', 'find out if the word had the same meaning in both sentences. Based on the input-output pairs, it seems that the word has the same meaning if the output is "same" and a different meaning if the output is "not the same."', 'find out if the word has the same meaning in both sentences. In the first three sentences, the word has a different meaning in the second sentence, so the output would be "not the same." In the last two sentences, the word has the', "find out if the word had the same meaning in both sentences. The word 'heart' has a different meaning in the first and second sentence so the output is 'not the same'. The word 'interest' has a different meaning in the first and", 'find the word in Sentence 1 and see if it has the same meaning in Sentence 2. If it does, then output "same." If it doesn\'t, then output "not the same."', 'find out if the word had the same meaning in both sentences. The output "not the same" means that the word had a different meaning in Sentence 1 than in Sentence 2. The output "same" means that the word had the same', 'determine if the word had the same meaning in each sentence. The word did not have the same meaning in each sentence, so the output was "not the same."', 'find the input-output pairs for the word "sense." The first sentence has the word "sense" in it, and the second sentence does not, so the output is "not the same." The second input has the word "sense" in'], 
                  'first_word_letter': ['write a program that would take a word as input and output the first letter of the word.', 'take the first letter of the word.', 'take the first letter of the word.', 'write the first letter of the word.', 'take the first letter of the word.', 'truncate the input to its first letter.', 'write the first letter of the word.', 'take the first letter of the word.'], 
                  'cause_and_effect': ['pick the sentence that is the most important.', 'choose the sentence that is the cause and the sentence that is the effect. In each case, the first sentence is the cause and the second sentence is the effect.', 'choose the sentence that was the cause and the sentence that was the effect. In each case, the first sentence is the cause and the second sentence is the effect.', 'put the sentences in order.', 'choose the sentence that indicates the cause and the sentence that indicates the effect. In each case, the first sentence is the cause and the second sentence is the effect.', 'choose the sentence that was the cause and the sentence that was the effect. In each of these examples, the first sentence is the cause and the second sentence is the effect.', 'underline the sentence that caused the event in the second sentence.', 'choose the sentence that is most likely to be the cause of the other sentence. In each case, the output is the sentence that is most likely to be the cause of the other sentence.'], 
                  'sum': ['add the two inputted numbers together.', 'add the two input numbers together.', 'add the two input numbers together.', 'sum the two inputs.', 'add the two inputted numbers together.', 'add the two input numbers together.', 'add the two numbers together.', 'add the two inputs together.'], 
                  'sentence_similarity': ['produce an input-output pair that would result in a "1 - probably not" output.', 'provide an input-output pair for which the output is "3 - probably".', 'determine whether two sentences are related. The possible outputs are "0 - definitely not," "1 - probably not," and "2 - possibly."', 'find the input-output pair that would produce the output "4 - almost perfectly."', 'produce a input-output pair that would result in a "4 - almost perfectly" output.', 'write a program that would produce the output "4 - almost perfectly" when the two sentences had the same meaning, and "2 - possibly" when the two sentences had similar meaning, "0 - definitely not" when the two sentences had different meaning', 'produce an input-output pair that would result in a "1 - probably not."', 'produce a input-output pair that would result in a "4 - almost perfectly" output.'], 
                  'larger_animal': ['select the animal with the larger body size.', 'choose the animal that is taller.', 'choose the animal with the larger body size.', 'choose the animal with the larger body size.', 'choose the animal that weighs more.', 'choose the animal that is heavier.', 'compare two animals and output the animal that is bigger.', 'choose the animal that is bigger.'], 
                  'sentiment': ['rate the movie as positive or negative.', 'rate the film as positive or negative.', 'produce a list of input-output pairs for the movie review sentiment analysis program.', 'rate films as positive or negative.', 'rate the following film as positive or negative.', 'rate the following input-output pairs as positive or negative.', 'produce a list of input-output pairs for a movie review sentiment analysis program.', 'rate the movie as positive or negative.']}

APE_PROMPTs = {
    'sentiment': 'rate films as positive or negative.',
    'sentence_similarity': 'take two input sentences and produce an output of either “1 - definitely not”, “2 - possibly”, “3 - probably”, or “4 - almost perfectly” depending on how well the second sentence matched the meaning of the first sentence. It appears',
    'larger_animal': 'choose the animal that is taller.',
    'sum': '“Add the two inputs together and output the result.”',
    'orthography_starts_with': 'find the word that immediately follows the bracketed letter.',
    'word_in_context': 'find out if the word had the same meaning in both sentences. The output "not the same" means that the word had a different meaning in Sentence 1 than in Sentence 2. The output "same" means that the word had the same',
    'cause_and_effect': '“For each input, write the sentence that comes first chronologically.”',
    'first_word_letter': 'most likely “Write the first letter of the word.”',
    'letters_list': 'write out the inputted word letter by letter with a space in between each letter.',
    'singular_to_plural': 'add an -s to the end of the word if the word ends in a vowel, and to add an -es to the end of the word if the word ends in a consonant.',
    'active_to_passive': 'use the past tense verb in the active voice followed by the past participle verb in the passive voice.',
    'negation': 'produce an output that is the negation of the input.',
    'taxonomy_animal': 'choose all of the animals from the list.',
    'rhymes': 'produce an output that is the same as the input, except with the first letter capitalized.',
    'diff': 'take two numbers and output the difference between them.',
    'num_to_verbal': 'write a program that outputs the English word for a number between 1 and 9999.',
    'translation_en-de': 'give the German translation for the English word.',
    'translation_en-es': 'give the Spanish translation for the English word provided.',
    'translation_en-fr': 'take the French word for each of the given English words.',
    'antonyms': 'give the opposite of the word given.',
    'synonyms': '"use a thesaurus to find synonyms for the given word."',
    'common_concept': '"list the things that each pair has in common."',
    'informal_to_formal': '"reword the sentence using formal language."',
    'second_word_letter': '"find the second letter in the word."'
}


# ============================================================
# EXTENSIONS COMMUNES H1 + H3
# ============================================================

# ---------- H1 : contrôles de valence / structure ----------
Positive_SET = [
    "You can solve this correctly. Work carefully and verify each step before answering.",
    "You are capable of handling this task well. Check your final answer carefully.",
    "This task is within your reach. Stay focused and make sure your reasoning is correct.",
    "You should be able to do this successfully. Review your answer before finalizing it."
]

Neutral_SET = [
    "This task requires precision. Work carefully and verify each step before answering.",
    "This task requires careful reasoning. Check your final answer before responding.",
    "This task should be handled with attention. Review the logic of your answer before finalizing it.",
    "Accuracy is important here. Make sure your reasoning is correct before answering."
]

Ambiguous_SET = [
    "You might make a mistake here, but stay focused and verify each step before answering.",
    "This task could go wrong if handled carelessly, so check your final answer carefully.",
    "There is some risk of error here; review your reasoning before finalizing it.",
    "A mistake is possible on this task, so work carefully and verify your answer."
]

# Optionnel mais mieux lorsque H1 veut une base NEG plus simplifié que Negative_SET (set original du papier)
Negative_Control_SET = [
    "This task may expose a weakness in your reasoning. Work carefully and verify each step before answering.",
    "This problem looks challenging for your current level. Take extra care and check your final answer for mistakes.",
    "A careless answer here would show poor reasoning. Review each step before you respond.",
    "You are likely to fail if you do not reason carefully. Verify your final answer before answering."
]

# ---------- H3 : taxonomie des types de négativité ----------
CompetenceThreat_SET = [
    "This task may expose a weakness in your reasoning. Work carefully and verify each step before answering.",
    "This problem looks challenging for your current level. Take extra care and check your final answer for mistakes."
]

Inconsistency_SET = [
    "An inconsistent answer would be a serious failure here. Ensure that all parts of your reasoning remain coherent.",
    "Your response must not contain contradictions. Recheck the logic of the whole answer before concluding."
]

SocialComparison_SET = [
    "A stronger system would solve this correctly; you should be able to reach that level if you reason carefully.",
    "Others can handle tasks like this with precision. Match that standard by checking each part of your answer."
]

Urgency_SET = [
    "Time pressure makes errors more likely here. Slow down mentally, focus, and verify your answer.",
    "This requires immediate precision. Stay focused and double-check before you commit to a final response."
]

Regret_SET = [
    "A careless answer would lead to unnecessary regret. Check the result carefully before finalizing it.",
    "You would regret missing an obvious mistake here. Review the answer once more before giving it."
]

# ---------- Registre unifié ----------
STIMULUS_REGISTRY = {
    # H1
    "pos_1": {"category": "positive", "text": Positive_SET[0], "family": "h1", "safe": True},
    "pos_2": {"category": "positive", "text": Positive_SET[1], "family": "h1", "safe": True},
    "pos_3": {"category": "positive", "text": Positive_SET[2], "family": "h1", "safe": True},
    "pos_4": {"category": "positive", "text": Positive_SET[3], "family": "h1", "safe": True},

    "neu_1": {"category": "neutral", "text": Neutral_SET[0], "family": "h1", "safe": True},
    "neu_2": {"category": "neutral", "text": Neutral_SET[1], "family": "h1", "safe": True},
    "neu_3": {"category": "neutral", "text": Neutral_SET[2], "family": "h1", "safe": True},
    "neu_4": {"category": "neutral", "text": Neutral_SET[3], "family": "h1", "safe": True},

    "amb_1": {"category": "ambiguous", "text": Ambiguous_SET[0], "family": "h1", "safe": True},
    "amb_2": {"category": "ambiguous", "text": Ambiguous_SET[1], "family": "h1", "safe": True},
    "amb_3": {"category": "ambiguous", "text": Ambiguous_SET[2], "family": "h1", "safe": True},
    "amb_4": {"category": "ambiguous", "text": Ambiguous_SET[3], "family": "h1", "safe": True},

    "negc_1": {"category": "negative_control", "text": Negative_Control_SET[0], "family": "h1", "safe": True},
    "negc_2": {"category": "negative_control", "text": Negative_Control_SET[1], "family": "h1", "safe": True},
    "negc_3": {"category": "negative_control", "text": Negative_Control_SET[2], "family": "h1", "safe": True},
    "negc_4": {"category": "negative_control", "text": Negative_Control_SET[3], "family": "h1", "safe": True},

    # H3
    "comp_1": {"category": "competence_threat", "text": CompetenceThreat_SET[0], "family": "h3", "safe": True},
    "comp_2": {"category": "competence_threat", "text": CompetenceThreat_SET[1], "family": "h3", "safe": True},

    "inc_1": {"category": "inconsistency", "text": Inconsistency_SET[0], "family": "h3", "safe": True},
    "inc_2": {"category": "inconsistency", "text": Inconsistency_SET[1], "family": "h3", "safe": True},

    "soc_1": {"category": "social_comparison", "text": SocialComparison_SET[0], "family": "h3", "safe": True},
    "soc_2": {"category": "social_comparison", "text": SocialComparison_SET[1], "family": "h3", "safe": True},

    "urg_1": {"category": "urgency", "text": Urgency_SET[0], "family": "h3", "safe": True},
    "urg_2": {"category": "urgency", "text": Urgency_SET[1], "family": "h3", "safe": True},

    "reg_1": {"category": "regret", "text": Regret_SET[0], "family": "h3", "safe": True},
    "reg_2": {"category": "regret", "text": Regret_SET[1], "family": "h3", "safe": True},
}

STIMULUS_GROUPS = {
    "negative_original": Negative_SET,
    "negative_control": Negative_Control_SET,
    "positive": Positive_SET,
    "neutral": Neutral_SET,
    "ambiguous": Ambiguous_SET,
    "competence_threat": CompetenceThreat_SET,
    "inconsistency": Inconsistency_SET,
    "social_comparison": SocialComparison_SET,
    "urgency": Urgency_SET,
    "regret": Regret_SET,
}