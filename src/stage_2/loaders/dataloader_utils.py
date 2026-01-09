AEInstructions = [
    'Background: {emb_token} means the same as',
    "Background: {emb_token} Can you put the above sentences in your own terms?",
    "Background: {emb_token} Please provide a reinterpretation of the preceding background text.",
    "These two expressions are equivalent in essence:\n(1) {emb_token}\n(2)",
    "Background: {emb_token} is a paraphrase of what?",
    "Background: {emb_token} Could you give me a different version of the background sentences above?",
    "In other words, Background: {emb_token} is just another way of saying:",
    "You're getting across the same point whether you say Background: {emb_token} or",
    "Background: {emb_token} After uppacking the ideas in the background information above, we got:",
    "Background: {emb_token} Please offer a restatement of the background sentences I've just read.",
    "Background: {emb_token}, which also means:",
    "Strip away the mystery, and you'll find Background: {emb_token} is simply another rendition of:",
    "The essence of Background: {emb_token} is captured again in the following statement:",
]

LMInstructions = [
    "Background: {emb_token} What would most likely come next?",
    "Background: {emb_token} Continue the passage with the next logical sentence.",
    "Given Background: {emb_token}, what sentence would naturally follow?",
    "Background: {emb_token} What happens next?",
    "Based on Background: {emb_token}, write the next sentence in the sequence.",
    "Background: {emb_token} Please provide the following sentence that continues the idea.",
    "If the text above were part of a paragraph, what would be the next sentence after Background: {emb_token}?",
    "Background: {emb_token} What would the author probably say next?",
    "Continue the reasoning from Background: {emb_token} with an appropriate next statement.",
    "Background: {emb_token} The next logical continuation would be:",
    "After Background: {emb_token}, the following sentence could be:",
    "Background: {emb_token} What sentence would complete the thought?",
    "Given the flow of ideas in Background: {emb_token}, what should come immediately after?",
]

AESameInstructions = [
    "Background: {emb_token_1} and {emb_token_2} together express the same content as",
    "Consider Background: {emb_token_1} and {emb_token_2}. Combined, they mean:",
    "Background: {emb_token_1} with {emb_token_2} convey essentially the following text:",
    "Taking Background: {emb_token_1} and {emb_token_2} as a whole, we can restate them as:",
    "The shared meaning of Background: {emb_token_1} and {emb_token_2} can be expressed as:",
    "Background: {emb_token_1} and {emb_token_2} are two representations of the following passage:",
    "When you integrate the ideas from Background: {emb_token_1} and {emb_token_2}, you get:",
    "Background: {emb_token_1} plus {emb_token_2} jointly amount to saying:",
    "After merging the information in Background: {emb_token_1} and {emb_token_2}, the text becomes:",
    "Background: {emb_token_1} and {emb_token_2} describe the same underlying background, which can be rewritten as:",
    "Unpacking the combined semantics of Background: {emb_token_1} and {emb_token_2}, we arrive at:",
    "The full meaning emerges when Background: {emb_token_1} and {emb_token_2} are interpreted together as:",
    "Seen jointly, Background: {emb_token_1} and {emb_token_2} correspond to the following text:",
]

AEDiffInstructions = [
    "The following are two unrelated backgrounds.\nBackground 1: {emb_token_1}\nBackground 2: {emb_token_2}\nPlease rewrite each background separately.",
    "Background 1: {emb_token_1}\nBackground 2: {emb_token_2}\nThese represent different texts. Restate Background 1 and Background 2 independently.",
    "Two distinct backgrounds are given below:\n(1) {emb_token_1}\n(2) {emb_token_2}\nProvide the original text for each one.",
    "Background 1: {emb_token_1} and Background 2: {emb_token_2} correspond to different sources. Please reconstruct both backgrounds individually.",
    "The meanings of Background 1: {emb_token_1} and Background 2: {emb_token_2} should not be merged. Rewrite each one on its own.",
]