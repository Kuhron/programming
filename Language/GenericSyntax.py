# define a way to input sentences by a morphosyntactic structure
# which can then be subject to movement, inflection, etc. depending on the language being translated to
# use this input format rather than English
# the input language should make all distinctions you would care to make; be specific with semantics and relations
# output will be based on a specific language object's parameters, vocabulary, allomorphy, etc.
# output will always be text but can represent spoken language, sign language, music, or whatever else
# in the case of music, can use MusicParser.py to create sound file from this
# will dramatically improve conlanging ability

# initial idea: use function syntax to represent trees

# the man eats a red apple
_sentence (
    _def (
        "man",
    ),
    _arg (
        _tma (
            "eat",
            "_present",
        ),
        _indef (
            _mod (
                "apple",
                "red",
            )
        )
    )
)

