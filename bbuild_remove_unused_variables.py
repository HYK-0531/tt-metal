import re
import logging


def remove_inline_comments(text):
    """Removes inline comments (//...) from a string."""
    return re.sub(r"//.*", "", text).rstrip()


def comment_out_unused_variables(log_file, num_variables_to_comment=5, output_log_file="build_removing_variables.log"):
    """
    Reads a build log, finds lines with "unused variable" warnings, and comments out the
    corresponding lines in the source files. Handles multi-line declarations, adds a comment flag,
    skips already commented out lines and for loops, and handles single-line multi-variable declarations.

    Args:
        log_file (str): Path to the build log file.
        num_variables_to_comment (int): The maximum number of unused variables to comment out.
        output_log_file (str): Path to the file where log messages will be written.
    """

    # Configure logging to write to a file
    logging.basicConfig(
        filename=output_log_file, filemode="w", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    count = 0
    with open(log_file, "r") as f:
        for line in f:
            if (
                "warning: unused variable" in line or "-Wunused-but-set-variable" in line
            ) and count < num_variables_to_comment:
                try:
                    # Extract filename and line number from the warning message
                    match = re.search(r"(.*?):(\d+):", line)
                    if not match:
                        logging.info(f"Could not parse line for filename and line number: {line.strip()}")
                        continue  # Skip to the next line

                    filename = match.group(1)
                    line_number = int(match.group(2))

                    # Read the entire source file
                    with open(filename, "r") as source_file:
                        source_lines = source_file.readlines()

                    if line_number <= 0 or line_number > len(source_lines):
                        logging.info(f"Line number {line_number} is out of range in file {filename}:{line_number}")
                        continue

                    # Get the problematic line
                    original_line = source_lines[line_number - 1].strip()

                    if (
                        line_number > 2
                        and source_lines[line_number - 2].strip() != ""
                        and (
                            (
                                remove_inline_comments(source_lines[line_number - 2].strip())[-1]
                                not in [";", "{", "}", "(", "[", "<"]
                            )
                            and ("//" not in source_lines[line_number - 2].strip()[0:2])
                        )
                    ):
                        logging.info(
                            f"Line {line_number} in {filename}:{line_number} is a continuation of a multi-line declaration. Skipping for now."
                        )
                        continue

                    if ";" not in original_line:
                        logging.info(f"Line {line_number} in {filename}:{line_number} is not a single line. Skipping.")
                        continue

                    # Extract the variable name from the warning
                    variable_name_match = re.search(r"variable '(\w+)'", line)
                    if not variable_name_match:
                        logging.info(f"Could not extract variable name from warning: {line.strip()}")
                        continue
                    variable_name = variable_name_match.group(1)

                    # Check if the line is already commented out
                    if "//" in original_line.lstrip()[0:2]:
                        logging.info(
                            f"Line {line_number} in {filename}:{line_number} is already commented out. Skipping."
                        )
                        continue

                    # check if the line is a for loop.
                    if "for(" in original_line.strip() or "for (" in original_line.strip():
                        logging.info(f"Line {line_number} in {filename}:{line_number} is a for loop. Skipping.")
                        continue

                    # Check for un-nested commas
                    def has_unnested_comma(text):
                        balance = {"(": 0, ")": 0, "[": 0, "]": 0, "{": 0, "}": 0, "<": 0, ">": 0}
                        for char in text:
                            if char in balance:
                                if char in "([{<":
                                    balance[char] += 1
                                else:
                                    balance[{")": "(", "]": "[", "}": "{", ">": "<"}[char]] -= 1
                            if char == "," and all(v == 0 for v in balance.values()):
                                return True
                        return False

                    if has_unnested_comma(original_line):
                        logging.info(
                            f"Line {line_number} in {filename}:{line_number} contains an un-nested comma. Skipping."
                        )
                        continue

                    # Add comment to the line
                    source_lines[line_number - 1] = (
                        " //" + source_lines[line_number - 1].rstrip() + " --commented out by unused variable remover\n"
                    )

                    # Write the modified content back to the source file
                    with open(filename, "w") as source_file:
                        source_file.writelines(source_lines)

                    logging.info(f"Added comment to line {line_number} in {filename}:{line_number}")
                    count += 1

                except FileNotFoundError:
                    logging.error(f"File not found: {filename}:{line_number}")
                except Exception as e:
                    logging.error(f"Error processing line: {line.strip()}. Error: {e}")

    logging.info(f"Commented out {count} unused variables.")


# Example usage:
log_file_path = "build.log"  # Replace with your actual log file path
number_of_variables = 999999999  # adjust to your needs
comment_out_unused_variables(log_file_path, number_of_variables)
