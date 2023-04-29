from sklearn.pipeline import Pipeline
from Pipeline_classes import *


# declareing a regex that contains some keywords to search for skincare issue
PATTERN = r"(skin|acne|wrinkle|dry|oil|dark|spot|red|age|aging|sag'|eye|bag|oily|\
             razor|bump|cold|sore|blist|hive|actinic|keratosis|rough|peel|peeling|eczema|rash|\
             chickenpox|measles|ringworm|rash|melasma|pore|measle|warts|cracked|crack|redness|sunburn|scrub|pimple\
             |dryness|enzyme|scar|hard)"

# this pipeline has the compelte preprocessing excluding removal of stopward and feature engineering
pipe_not_completely_cleaned = Pipeline(
    steps=[
        ("convert_to_lower_case", ConvertToLower()),
        ("remove_special_caharacter", RemoveSpecicialCharacter()),
        ("lemmertize", Lemmertize()),
        ("Create Sentiment Score", CreateSentimentScore()),
        ("Create Rating(Three)", CreateRatingThree()),
        ("Create Rating((Two)", CreateRatingTwo()),
        ("Create Skin Issue", CreateSkinIssue(pattern=PATTERN)),
    ]
)

# this pipeline has the compelte preprocessing including removal of stopward and feature engineering
pipe_completely_cleaned = Pipeline(
    steps=[
        ("convert_to_lower_case", ConvertToLower()),
        ("remove_stop_word", RemoveStopwords()),
        ("remove_special_caharacter", RemoveSpecicialCharacter()),
        ("lemmertize", Lemmertize()),
        ("Create Sentiment Score", CreateSentimentScore()),
        ("Create Rating(Three)", CreateRatingThree()),
        ("Create Rating((Two)", CreateRatingTwo()),
        ("Create Skin Issue", CreateSkinIssue(pattern=PATTERN)),
    ]
)