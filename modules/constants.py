"""
Constants and configuration values for the Virtual Makeup Application
"""

# MediaPipe Face Mesh landmark indices for lips
# Upper outer lip
UPPER_OUTER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409]
# Lower outer lip
LOWER_OUTER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
# Upper inner lip
UPPER_INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415]
# Lower inner lip
LOWER_INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]

# Combined lip indices
LIPS_ALL = UPPER_OUTER_LIP + LOWER_OUTER_LIP
INNER_LIPS = UPPER_INNER_LIP + LOWER_INNER_LIP

# Lipstick color presets (in BGR format)
LIPSTICK_COLORS = {
    "Red": (43, 43, 200),
    "Pink": (147, 20, 255),
    "Coral": (114, 128, 250),
    "Nude": (169, 184, 215),
    "Burgundy": (43, 0, 125),
    "Purple": (164, 73, 163),
    "Orange": (58, 127, 240)
}

