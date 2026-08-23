"""AUTHORITATIVE move-template key triples.

Generated from the WOTR-Gandalf app's messages.properties
(the one-off generator was retired in the session-36
cleanup; regeneration means re-extracting the numeric
triple templates from messages.properties).  DO NOT
hand-edit the data.  Each triple (k1, k2, k3) encodes a
piece-move log line
    <~k1~>pieces <~k2~><~from~><~k3~><~to~>
SUFFIXED_TRIPLES carry a baked-in destination in the k3
text (no location code follows).  Code consumers import
only MOVE_TRIPLES; the CardBuffet.9/14 play-area handling
hardcodes its keys, so SUFFIXED_TRIPLES is the reference
record of that source, not a live input.
"""

MOVE_TRIPLES = [
    ('CardBuffet.3', 'CardBuffet.4', 'CardBuffet.5'),
    ('Controls.271', 'Controls.272', 'Controls.273'),
    ('Controls.276', 'Controls.277', 'Controls.278'),
    ('Controls.281', 'Controls.282', 'Controls.283'),
    ('Controls.358', 'Controls.359', 'Controls.360'),
    ('Controls.366', 'Controls.367', 'Controls.368'),
    ('Controls.534', 'Controls.535', 'Controls.536'),
    ('Controls.539', 'Controls.540', 'Controls.541'),
    ('Controls.573', 'Controls.574', 'Controls.575'),
    ('Controls.578', 'Controls.579', 'Controls.580'),
    ('Controls.650', 'Controls.651', 'Controls.652'),
    ('Controls.655', 'Controls.656', 'Controls.657'),
    ('Controls.726', 'Controls.727', 'Controls.728'),
    ('Controls.731', 'Controls.732', 'Controls.733'),
    ('Controls.736', 'Controls.737', 'Controls.738'),
    ('Controls.743', 'Controls.744', 'Controls.745'),
    ('Controls.750', 'Controls.751', 'Controls.752'),
    ('Controls.757', 'Controls.758', 'Controls.759'),
    ('Controls.762', 'Controls.763', 'Controls.764'),
    ('Controls.767', 'Controls.768', 'Controls.769'),
    ('Controls.779', 'Controls.780', 'Controls.781'),
    ('Controls.784', 'Controls.785', 'Controls.786'),
    ('Controls.790', 'Controls.791', 'Controls.792'),
    ('Controls.795', 'Controls.796', 'Controls.797'),
    ('Controls.802', 'Controls.803', 'Controls.804'),
    ('Controls.811', 'Controls.812', 'Controls.813'),
    ('DiceBuffet.1', 'DiceBuffet.2', 'DiceBuffet.3'),
    ('HighlightLabel.81', 'HighlightLabel.82', 'HighlightLabel.83'),
    ('HighlightLabel.86', 'HighlightLabel.87', 'HighlightLabel.88'),
    ('HighlightLabel.91', 'HighlightLabel.92', 'HighlightLabel.93'),
    ('HighlightLabel.96', 'HighlightLabel.97', 'HighlightLabel.98'),
    ('HighlightLabel.101', 'HighlightLabel.102', 'HighlightLabel.103'),
    ('Interpreter.105', 'Interpreter.106', 'Interpreter.107'),
    ('Scribe.7', 'Scribe.8', 'Scribe.9'),
    ('Scribe.14', 'Scribe.15', 'Scribe.16'),
    ('Scribe.17', 'Scribe.18', 'Scribe.19'),
    ('Scribe.24', 'Scribe.25', 'Scribe.26'),
    ('Scribe.33', 'Scribe.34', 'Scribe.35'),
    ('Scribe.40', 'Scribe.41', 'Scribe.42'),
    ('Scribe.45', 'Scribe.46', 'Scribe.47'),
    ('Scribe.50', 'Scribe.51', 'Scribe.52'),
    ('Scribe.54', 'Scribe.55', 'Scribe.56'),
    ('Scribe.58', 'Scribe.59', 'Scribe.60'),
    ('Scribe.65', 'Scribe.66', 'Scribe.67'),
    ('Scribe.69', 'Scribe.70', 'Scribe.71'),
    ('Scribe.73', 'Scribe.74', 'Scribe.75'),
    ('Scribe.77', 'Scribe.78', 'Scribe.79'),
    ('Scribe.85', 'Scribe.86', 'Scribe.87'),
    ('Scribe.88', 'Scribe.89', 'Scribe.90'),
    ('Scribe.141', 'Scribe.142', 'Scribe.143'),
]

SUFFIXED_TRIPLES = [
    ('CardBuffet.9', 'CardBuffet.10', 'CardBuffet.11', 'to current fp card'),
    ('CardBuffet.14', 'CardBuffet.15', 'CardBuffet.16', 'to current sa card'),
]
