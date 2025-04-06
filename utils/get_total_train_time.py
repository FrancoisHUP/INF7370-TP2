import re

def parse_time_from_line(line):
    """
    Extracts all occurrences of time in the format '<seconds>s <milliseconds>ms/step'
    from a single line and returns the cumulative time in seconds.
    """
    total_seconds = 0.0
    # Find all occurrences of the pattern: one or more digits followed by 's',
    # a space, one or more digits followed by 'ms/step'
    matches = re.findall(r'(\d+)s (\d+)ms/step', line)
    for match in matches:
        seconds = int(match[0])
        milliseconds = int(match[1])
        total_seconds += seconds + milliseconds / 1000.0
    return total_seconds

def calculate_total_time(log_filename):
    """
    Reads the log file, extracts time stamps from each line, and calculates the total time.
    """
    total_time = 0.0
    with open(log_filename, 'r') as file:
        for line in file:
            total_time += parse_time_from_line(line)
    return total_time

if __name__ == "__main__":
    log_file = "output/10_deep_wide/trace.txt"  # Replace with your actual log file name
    total_training_time = calculate_total_time(log_file)
    print(f"Total training time: {total_training_time:.2f} seconds")
