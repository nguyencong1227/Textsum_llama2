from termcolor import colored

def format_wonder_city(wonder_city):
    return colored(wonder_city, "green", attrs=["bold", "underline"])

def format_summary(summary):
    return colored(f"Summary: {summary}", "black")

def display_result(wonder_city, summary):
    formatted_wonder_city = format_wonder_city(wonder_city)
    formatted_summary = format_summary(summary)

    print(formatted_wonder_city)
    print()
    print(formatted_summary)
    print("\n----------------------------------------------\n")
