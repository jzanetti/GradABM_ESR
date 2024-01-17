DIARY_CFG = {
    "worker": {
        "household": {"weight": 0.25, "time_ranges": [(0, 8), (15, 24)]},
        "travel": {"weight": 0.15, "time_ranges": [(7, 9), (16, 19)]},
        "company": {"weight": 0.4, "time_ranges": [(9, 17)]},
        "supermarket": {"weight": 0.1, "time_ranges": [(8, 9), (17, 20)]},
        "restaurant": {"weight": 0.1, "time_ranges": [(7, 8), (18, 20)]},
    },
    "student": {
        "household": {"weight": 0.3, "time_ranges": [(0, 8), (15, 24)]},
        "school": {"weight": 0.5, "time_ranges": [(9, 15)]},
        "supermarket": {"weight": 0.1, "time_ranges": [(8, 9), (17, 20)]},
        "restaurant": {"weight": 0.1, "time_ranges": [(7, 8), (18, 20)]},
    },
    "default": {
        "household": {"weight": 0.6, "time_ranges": [(0, 24)]},
        "supermarket": {"weight": 0.2, "time_ranges": [(8, 9), (17, 20)]},
        "restaurant": {"weight": 0.2, "time_ranges": [(7, 8), (18, 20)]},
    },
}
